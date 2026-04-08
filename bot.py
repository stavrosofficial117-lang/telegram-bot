"""
Telegram Personal Assistant Bot — powered by Claude AI
"""

import os
import logging
import asyncio
import re
import json
import subprocess
import tempfile
from pathlib import Path
from telegram import Update, Document
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from telegram.constants import ChatAction
import anthropic

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")   # optional whitelist

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# Per-user conversation memory  {user_id: [{"role": ..., "content": ...}]}
conversations: dict[int, list] = {}
MAX_HISTORY = 40          # messages kept per user

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a powerful personal AI assistant delivered through Telegram.
You can help with ANYTHING the user asks, including:

• Writing & editing (emails, essays, code, scripts, reports)
• Building websites / HTML pages (provide full source code)
• Coding in any language (Python, JS, bash, SQL, …)
• Data analysis, math, planning, research, brainstorming
• Answering questions on any topic
• Creating files (output their content clearly in code blocks)
• Step-by-step instructions for complex tasks
• Creative work (stories, poems, marketing copy)
• Summarising documents the user pastes

Guidelines:
- When you produce a file (HTML, Python, etc.) wrap it in a fenced code block with the language tag.
- For websites provide complete, self-contained HTML/CSS/JS in one block.
- Be concise but thorough. Use markdown formatting.
- If a task is multi-step, number the steps.
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
    """Return list of (language, code) tuples from fenced blocks."""
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

async def send_long_message(update: Update, text: str):
    """Telegram max is 4096 chars — split if needed."""
    max_len = 4000
    for i in range(0, len(text), max_len):
        await update.message.reply_text(
            text[i : i + max_len],
            parse_mode="Markdown"
        )

async def ask_claude(user_id: int, user_message: str) -> str:
    history = conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    conversations[user_id] = history

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})
    conversations[user_id] = history
    return assistant_text

# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Unauthorised.")
        return
    await update.message.reply_text(
        f"👋 Hi *{user.first_name}*! I'm your personal AI assistant powered by Claude.\n\n"
        "Ask me *anything* — write code, build websites, answer questions, draft emails, "
        "analyse data, and much more.\n\n"
        "Commands:\n"
        "/start — this message\n"
        "/clear — reset conversation memory\n"
        "/help  — show examples\n",
        parse_mode="Markdown"
    )

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    conversations.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Conversation cleared. Fresh start!")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "*Things you can ask me:*\n\n"
        "🌐 `Build me a landing page for a coffee shop`\n"
        "🐍 `Write a Python script to rename files in a folder`\n"
        "📧 `Draft a professional follow-up email`\n"
        "📊 `Explain how to set up a PostgreSQL database`\n"
        "🎨 `Create a CSS card component with hover animation`\n"
        "📝 `Summarise this text: [paste text]`\n"
        "🔢 `Solve this math problem step by step`\n"
        "🤖 `Write a bash script to back up my files`\n\n"
        "Just type naturally — I'll figure out what you need!",
        parse_mode="Markdown"
    )

# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ Unauthorised.")
        return

    user_text = update.message.text or ""
    if not user_text.strip():
        return

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    try:
        reply = await ask_claude(user.id, user_text)
    except Exception as e:
        logger.error("Claude error: %s", e)
        await update.message.reply_text(f"⚠️ Error talking to Claude: {e}")
        return

    # Send the text reply
    await send_long_message(update, reply)

    # Auto-send any code blocks as downloadable files
    blocks = extract_code_blocks(reply)
    if blocks:
        for idx, (lang, code) in enumerate(blocks):
            if len(code.strip()) > 80 and lang:          # only non-trivial blocks
                file_path = save_code_block(lang, code, idx)
                await update.message.reply_document(
                    document=open(file_path, "rb"),
                    filename=file_path.name,
                    caption=f"📄 `{file_path.name}`",
                    parse_mode="Markdown"
                )

# ── Document handler (user sends a file to analyse) ───────────────────────────

async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        return

    doc: Document = update.message.document
    caption = update.message.caption or "Please analyse or summarise this file."

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

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
        await update.message.reply_text(f"⚠️ Error: {e}")
        return

    await send_long_message(update, reply)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot is running…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
