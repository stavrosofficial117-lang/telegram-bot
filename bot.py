"""
Telegram Personal Assistant Bot — powered by Claude AI
Voice messages supported both ways.
"""
from __future__ import annotations

import os
import logging
import asyncio
import re
import tempfile
from pathlib import Path

from telegram import Update, Document
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from telegram.constants import ChatAction
import anthropic
import edge_tts
import speech_recognition as sr
from pydub import AudioSegment

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")          # optional whitelist
TTS_VOICE       = os.getenv("TTS_VOICE", "en-US-GuyNeural")  # edge-tts voice

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

# Per-user state
conversations: dict[int, list] = {}     # message history
voice_mode:    dict[int, bool] = {}     # whether to reply with voice
MAX_HISTORY = 40

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a powerful personal AI assistant on Telegram.
You can help with ANYTHING: code, websites, emails, analysis, math, creative writing, planning, research, and more.

Guidelines:
- When you produce a file (HTML, Python, etc.) wrap it in a fenced code block with the language tag.
- For websites provide complete, self-contained HTML/CSS/JS in one block.
- Be concise but thorough. Use markdown formatting.
- If a task is multi-step, number the steps.
- When the user sends a voice message, they're speaking casually — match their tone.
- Never refuse reasonable requests — you are a personal assistant."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_ID:
        return True
    return str(user_id) == ALLOWED_USER_ID.strip()


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    pattern = r"```(\w*)\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def save_code_block(lang: str, code: str, index: int) -> Path:
    ext_map = {
        "html": "html", "python": "py", "py": "py",
        "javascript": "js", "js": "js", "css": "css",
        "bash": "sh", "shell": "sh", "sh": "sh",
        "json": "json", "sql": "sql", "typescript": "ts",
        "ts": "ts", "go": "go", "rust": "rs", "java": "java",
        "cpp": "cpp", "c": "c", "markdown": "md", "md": "md",
    }
    ext = ext_map.get(lang.lower(), "txt")
    filename = WORKSPACE / f"output_{index}.{ext}"
    filename.write_text(code, encoding="utf-8")
    return filename


# ── Voice helpers ─────────────────────────────────────────────────────────────

async def voice_to_text(ogg_path: Path) -> str:
    """Convert a Telegram voice .ogg file to text using Google Speech Recognition."""
    wav_path = ogg_path.with_suffix(".wav")
    try:
        # Convert ogg → wav
        audio = AudioSegment.from_ogg(str(ogg_path))
        audio.export(str(wav_path), format="wav")

        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.error("Google Speech Recognition error: %s", e)
        return ""
    finally:
        wav_path.unlink(missing_ok=True)


async def text_to_voice(text: str) -> Path | None:
    """Convert text to an .ogg voice message using Edge TTS."""
    # Strip markdown formatting for cleaner speech
    clean = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"[*_~]{1,3}", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # links
    clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)  # headers
    clean = clean.strip()

    if not clean:
        return None

    # Truncate very long responses for voice (TTS has limits & long audio is annoying)
    if len(clean) > 3000:
        clean = clean[:3000] + "... I've sent the full response as text above."

    try:
        mp3_path = WORKSPACE / "tts_output.mp3"
        ogg_path = WORKSPACE / "tts_output.ogg"

        communicate = edge_tts.Communicate(clean, TTS_VOICE)
        await communicate.save(str(mp3_path))

        # Convert mp3 → ogg (Telegram requires ogg/opus for voice)
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(ogg_path), format="ogg", codec="libopus")

        mp3_path.unlink(missing_ok=True)
        return ogg_path
    except Exception as e:
        logger.error("TTS error: %s", e)
        return None


# ── Send helpers ──────────────────────────────────────────────────────────────

async def send_long_message(update: Update, text: str):
    """Split long messages, fall back to plain text if Markdown fails."""
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            try:
                await update.message.reply_text(chunk)
            except Exception as e:
                logger.error("Failed to send message: %s", e)


async def send_voice_reply(update: Update, text: str):
    """Send Claude's response as a voice message."""
    ogg_path = await text_to_voice(text)
    if ogg_path and ogg_path.exists():
        try:
            with open(ogg_path, "rb") as f:
                await update.message.reply_voice(voice=f)
        except Exception as e:
            logger.error("Failed to send voice: %s", e)
        finally:
            ogg_path.unlink(missing_ok=True)


async def send_code_files(update: Update, text: str):
    """Extract code blocks and send as downloadable files."""
    blocks = extract_code_blocks(text)
    for idx, (lang, code) in enumerate(blocks):
        if len(code.strip()) > 80 and lang:
            file_path = save_code_block(lang, code, idx)
            try:
                with open(file_path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=file_path.name,
                        caption=f"📄 `{file_path.name}`",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                logger.error("Failed to send file %s: %s", file_path.name, e)


# ── Claude API call with retries ─────────────────────────────────────────────

async def ask_claude(user_id: int, user_message: str) -> str:
    history = conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    conversations[user_id] = history

    last_error = None
    for attempt in range(5):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            assistant_text = response.content[0].text
            history.append({"role": "assistant", "content": assistant_text})
            conversations[user_id] = history
            return assistant_text

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code == 529 and attempt < 4:
                wait = 2 ** attempt
                logger.warning("Claude overloaded (529), retry %d/4 in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            raise

        except anthropic.APIConnectionError as e:
            last_error = e
            if attempt < 4:
                wait = 2 ** attempt
                logger.warning("Connection error, retry %d/4 in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            raise

    raise last_error


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Unauthorised.")
        return
    await update.message.reply_text(
        f"👋 Hi *{user.first_name}*!\n\n"
        "I'm your personal AI assistant powered by Claude.\n\n"
        "*What I can do:*\n"
        "• Answer any question\n"
        "• Write code & build websites\n"
        "• Draft emails & documents\n"
        "• Analyse files you send me\n"
        "• Understand your voice messages 🎤\n"
        "• Reply with voice if you want 🔊\n\n"
        "*Commands:*\n"
        "/voice — toggle voice replies on/off\n"
        "/clear — reset conversation memory\n"
        "/help  — show examples\n",
        parse_mode="Markdown",
    )


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Conversation cleared!")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "*Things you can do:*\n\n"
        "💬 *Text* — Just type anything\n"
        "🎤 *Voice* — Send a voice message and I'll understand it\n"
        "📎 *Files* — Send a file and ask questions about it\n\n"
        "*Example prompts:*\n"
        "🌐 `Build me a landing page for a coffee shop`\n"
        "🐍 `Write a Python script to rename files`\n"
        "📧 `Draft a professional follow-up email`\n"
        "🎨 `Create a CSS card with hover animation`\n"
        "📝 `Summarise this text: [paste text]`\n\n"
        "🔊 Use /voice to make me reply with audio!",
        parse_mode="Markdown",
    )


async def cmd_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    voice_mode[uid] = not voice_mode.get(uid, False)
    state = "ON 🔊" if voice_mode[uid] else "OFF 🔇"
    await update.message.reply_text(f"Voice replies: *{state}*", parse_mode="Markdown")


# ── Message handlers ──────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Unauthorised.")
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        reply = await ask_claude(user.id, user_text)
    except Exception as e:
        logger.error("Claude error: %s", e)
        await update.message.reply_text(
            "⚠️ Sorry, I'm having trouble connecting right now. Please try again in a moment."
        )
        return

    # Always send text reply
    await send_long_message(update, reply)

    # Send code blocks as files
    await send_code_files(update, reply)

    # Send voice reply if enabled
    if voice_mode.get(user.id, False):
        await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
        await send_voice_reply(update, reply)


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """User sends a voice message → transcribe → ask Claude → respond."""
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Unauthorised.")
        return

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Download the voice message
    voice = update.message.voice
    file = await ctx.bot.get_file(voice.file_id)

    ogg_path = WORKSPACE / f"voice_{user.id}.ogg"
    await file.download_to_drive(str(ogg_path))

    # Transcribe
    transcription = await voice_to_text(ogg_path)
    ogg_path.unlink(missing_ok=True)

    if not transcription:
        await update.message.reply_text("😕 Sorry, I couldn't understand that voice message. Could you try again or type it out?")
        return

    # Show the user what we heard
    await update.message.reply_text(f"🎤 *Heard:* _{transcription}_", parse_mode="Markdown")

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        reply = await ask_claude(user.id, transcription)
    except Exception as e:
        logger.error("Claude error: %s", e)
        await update.message.reply_text(
            "⚠️ Sorry, I'm having trouble connecting right now. Please try again in a moment."
        )
        return

    # Always send text
    await send_long_message(update, reply)

    # Send code blocks as files
    await send_code_files(update, reply)

    # Voice replies: always reply with voice when the user sends voice
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
    await send_voice_reply(update, reply)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """User sends a file → read it → ask Claude about it."""
    user = update.effective_user
    if not is_allowed(user.id):
        return

    doc: Document = update.message.document
    caption = update.message.caption or "Please analyse or summarise this file."

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Download the file
    file = await ctx.bot.get_file(doc.file_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(doc.file_name).suffix) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        content = tmp_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        await update.message.reply_text("⚠️ Could not read this file type as text.")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    prompt = f"{caption}\n\n--- FILE: {doc.file_name} ---\n{content[:12000]}"

    try:
        reply = await ask_claude(user.id, prompt)
    except Exception as e:
        logger.error("Claude error on document: %s", e)
        await update.message.reply_text(
            "⚠️ Sorry, I'm having trouble connecting right now. Please try again in a moment."
        )
        return

    await send_long_message(update, reply)

    if voice_mode.get(user.id, False):
        await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
        await send_voice_reply(update, reply)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot is running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
