import asyncio
import logging
import time
import os
import base64
import tempfile
import aiohttp
from datetime import datetime
from functools import wraps
from pathlib import Path

# Telegram
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from telegram.constants import ChatAction

# Anthropic (Claude)
import anthropic

# Groq (Whisper voice transcription - free & fast)
from groq import AsyncGroq

# Text-to-Speech
import edge_tts

# Audio conversion
from pydub import AudioSegment

# Tavily web search
from tavily import TavilyClient

# Database, project builder & memory engine
from database_manager import db
from project_builder import builder
from memory_engine import memory_engine

# Environment
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

BOT_TOKEN         = os.getenv("BOT_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY    = None  # Not needed — using Groq
TTS_VOICE         = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
WEATHER_API_KEY   = os.getenv("WEATHER_API_KEY")
_raw_id           = os.getenv("ALLOWED_USER_ID", "")
ALLOWED_USER_ID   = int(_raw_id) if _raw_id.strip().isdigit() else None

# Supported text file extensions for file reading
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".html", ".css", ".json",
    ".md", ".txt", ".csv", ".yaml", ".yml", ".env",
    ".sh", ".sql", ".xml", ".toml", ".ini", ".cfg"
}

# Track builds waiting for user answers {user_id: original_description}
pending_builds: dict = {}

# Keywords that trigger the project builder naturally
BUILD_TRIGGERS = [
    "build me", "build a", "create a project", "make me a",
    "generate a project", "write me a project", "develop a",
    "code me a", "create me a"
]

# Keywords that trigger a web search
SEARCH_TRIGGERS = [
    "search for", "search ", "look up", "find me", "what's the latest",
    "latest news", "current price", "what is happening", "today's",
    "news about", "recent news", "right now", "currently", "live price",
    "stock price", "weather in", "who won", "what happened to"
]

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  API CLIENTS
# ─────────────────────────────────────────────

claude      = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
tavily      = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# ─────────────────────────────────────────────
#  RATE LIMITING
# ─────────────────────────────────────────────

user_last_message: dict = {}
RATE_LIMIT = 1  # seconds between messages per user

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a powerful personal AI dev assistant on Telegram.
You help with code, full-stack projects, emails, research, analysis, and more.

Guidelines:
- When producing a file (HTML, Python, etc.) wrap it in a fenced code block with the language tag.
- For websites provide complete, self-contained HTML/CSS/JS in one block.
- Be concise but thorough. Use markdown formatting.
- If a task is multi-step, number the steps clearly.
- Match the user's tone — casual or professional.
- When given a file or image, analyse it and answer the user's question about it.
- Never refuse reasonable requests."""

# ─────────────────────────────────────────────
#  DECORATORS
# ─────────────────────────────────────────────

def private_only(func):
    """Restrict bot to ALLOWED_USER_ID if set."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if ALLOWED_USER_ID and update.effective_user.id != ALLOWED_USER_ID:
            await update.message.reply_text("⛔ This is a private bot.")
            return
        return await func(update, context)
    return wrapper


def rate_limit(func):
    """Allow one message per RATE_LIMIT seconds per user."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        now = time.time()
        if now - user_last_message.get(user_id, 0) < RATE_LIMIT:
            await update.message.reply_text(
                "⏰ Please wait a moment before sending another message."
            )
            return
        user_last_message[user_id] = now
        return await func(update, context)
    return wrapper

# ─────────────────────────────────────────────
#  VOICE HELPERS
# ─────────────────────────────────────────────

async def transcribe_voice(ogg_path: str) -> str:
    """Transcribe a voice message using Groq Whisper API."""
    with open(ogg_path, "rb") as audio_file:
        transcript = await groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file
        )
    return transcript.text


async def text_to_voice(text: str, out_path: str):
    """Convert text to speech using edge-tts and save as mp3."""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(out_path)


async def send_long_message(update: Update, text: str):
    """Send a message, splitting it if it exceeds Telegram's 4096 char limit."""
    max_length = 4000
    if len(text) <= max_length:
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)
        return

    # Split into chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(chunk)


async def strip_markdown(text: str) -> str:
    """Remove markdown formatting before sending to TTS."""
    import re
    text = re.sub(r'#{1,6}\s*', '', text)        # headers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # italic
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)  # code
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # links
    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'\n{2,}', '\n', text)          # extra newlines
    return text.strip()


async def send_voice_reply(update: Update, text: str):
    """Generate TTS and send as a Telegram voice message."""
    mp3_path = tempfile.mktemp(suffix=".mp3")
    ogg_path = tempfile.mktemp(suffix=".ogg")

    try:
        clean_text = await strip_markdown(text)
        await text_to_voice(clean_text, mp3_path)
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(ogg_path, format="ogg", codec="libopus")

        with open(ogg_path, "rb") as voice_file:
            await update.message.reply_voice(voice=voice_file)

    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Silently fail — text reply already sent

    finally:
        for path in [mp3_path, ogg_path]:
            if os.path.exists(path):
                os.remove(path)

# ─────────────────────────────────────────────
#  AI RESPONSE
# ─────────────────────────────────────────────

async def web_search(query: str) -> str:
    """Search the web using Tavily and return formatted results."""
    if not tavily:
        return ""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: tavily.search(query=query, max_results=5)
        )
        if not results or "results" not in results:
            return ""

        formatted = f"🔍 Web search results for: *{query}*\n\n"
        for i, r in enumerate(results["results"], 1):
            formatted += f"{i}. **{r.get('title', 'No title')}**\n"
            formatted += f"   {r.get('content', '')[:200]}...\n"
            formatted += f"   🔗 {r.get('url', '')}\n\n"
        return formatted
    except Exception as e:
        logger.error(f"Search error: {e}")
        return ""


MEMORY_EXTRACTION_PROMPT = """You are a memory extraction assistant.
Given a conversation message, extract any important personal facts worth remembering.
These include: preferences, skills, projects, goals, constraints, personal details.

Respond with a JSON array of strings, each being a concise memory to save.
If nothing important, respond with an empty array: []

Examples:
- "I prefer Python" -> ["User prefers Python over other languages"]
- "I'm building a SaaS" -> ["User is building a SaaS product"]
- "my budget is $500" -> ["User's budget is $500"]
- "what's 2+2" -> []

Respond ONLY with the JSON array, no other text."""


async def process_memories(user_id: int, message: str):
    """Extract, save and summarize memories from a message."""
    try:
        # Extract new memories
        new_memories = await memory_engine.extract_memories(message)
        for item in new_memories:
            await db.save_memory(
                user_id,
                item.get("memory", ""),
                item.get("category", "general"),
                item.get("importance", 1)
            )

        # Update summary every 5 messages
        all_memories = await db.get_memories(user_id, limit=50)
        if all_memories and len(all_memories) % 5 == 0:
            summary = await memory_engine.summarize_memories(all_memories)
            if summary:
                await db.save_memory_summary(user_id, summary)

    except Exception as e:
        logger.error(f"Memory processing error: {e}")


async def get_ai_response(user_message: str, user_id: int,
                           extra_content: list = None) -> str:
    """Get a response from Claude with Level 3 persistent memory."""
    history = await db.get_conversation_history(user_id, limit=10)

    # Load memory summary and relevant memories
    summary = await db.get_memory_summary(user_id)
    all_memories = await db.get_memories(user_id, limit=50)
    relevant = await memory_engine.get_relevant_memories(user_message, all_memories)
    stats = await db.get_user_stats(user_id)

    # Build rich memory context
    memory_context = memory_engine.build_memory_context(summary, relevant, stats)
    enhanced_system = SYSTEM_PROMPT + memory_context

    if extra_content:
        content = extra_content + [{"type": "text", "text": user_message}]
        history.append({"role": "user", "content": content})
    else:
        history.append({"role": "user", "content": user_message})

    response = await claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=enhanced_system,
        messages=history
    )

    ai_response = response.content[0].text
    await db.log_conversation(user_id, user_message, ai_response)

    # Process memories in background
    asyncio.create_task(process_memories(user_id, user_message))

    return ai_response

async def read_url(url: str) -> str:
    """Fetch and extract text content from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return f"Could not fetch URL (status {resp.status})"
                html = await resp.text()

        # Strip HTML tags simply
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate to 8000 chars
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"
        return text
    except Exception as e:
        return f"Error reading URL: {e}"


async def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    if not WEATHER_API_KEY:
        # Fall back to Tavily search
        results = await web_search(f"current weather in {city} today")
        return results or f"No weather data available for {city}"

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()

        if data.get("cod") != 200:
            return f"City not found: {city}"

        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        desc = data["weather"][0]["description"].capitalize()
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        city_name = data["name"]
        country = data["sys"]["country"]

        return (
            f"🌤️ *Weather in {city_name}, {country}*\n\n"
            f"🌡️ Temperature: {temp}°C (feels like {feels}°C)\n"
            f"☁️ Condition: {desc}\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind: {wind} m/s"
        )
    except Exception as e:
        return f"Weather error: {e}"


async def generate_image(prompt: str) -> bytes:
    """Generate an image using SDXL via Replicate."""
    if not REPLICATE_API_KEY:
        return None
    try:
        import replicate
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={"prompt": prompt, "num_outputs": 1, "width": 1024, "height": 1024}
            )
        )
        if output and len(output) > 0:
            file_output = output[0]
            if hasattr(file_output, "read"):
                return file_output.read()
            url = str(file_output)
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.read()
        return None
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None


async def generate_briefing(user_id: int) -> str:
    """Generate a daily briefing with news + reminders + summary."""
    # Get reminders from memory
    memories = await db.get_memories_by_category(user_id, "reminder")
    reminders_text = ""
    if memories:
        reminders_text = "📌 *Your Reminders:*\n"
        reminders_text += "\n".join(f"• {m.replace('REMINDER: ', '')}" for m in memories)
        reminders_text += "\n\n"

    # Get top news
    news = await web_search("top news today")

    # Get weather for Irvine by default (from memory)
    weather = await get_weather("Irvine")

    # Ask Claude to compile everything
    prompt = (
        f"Create a concise morning briefing. Include:\n\n"
        f"Weather:\n{weather}\n\n"
        f"Reminders:\n{reminders_text or 'None'}\n\n"
        f"Top News:\n{news}\n\n"
        f"Keep it punchy and useful. Use emojis."
    )

    briefing = await get_ai_response(prompt, user_id)
    return briefing


# ─────────────────────────────────────────────
#  COMMANDS
# ─────────────────────────────────────────────

@private_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await db.add_or_update_user(
        user.id, user.username or "",
        user.first_name or "", user.last_name
    )
    await update.message.reply_text("""
🤖 *Welcome to your Personal AI Dev Assistant!*

Powered by Claude — your private dev partner on Telegram.

*What I can do:*
• 💻 Write & debug code in any language
• 🌐 Build complete websites & web apps
• 📎 Read & analyse files and images you send
• 🎤 Understand voice messages
• 🔊 Reply with voice (toggle with /voice)
• 📦 Build full projects and send as `.zip`
• 🧠 Remember our conversations

*Commands:*
/start — Welcome message
/help — Full help & examples
/voice — Toggle voice replies on/off
/build — Build a full project
/clear — Clear conversation memory
/stats — Your usage statistics

Just send me a message to get started! 🚀
    """, parse_mode="Markdown")


@private_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
🆘 *Help & Examples*

*Commands:*
/start — Welcome
/help — This message
/voice — Toggle voice replies 🔊
/build `<description>` — Build a full project as .zip
/clear — Reset conversation memory
/stats — Usage statistics

*Examples — just type:*
• "Write a Python web scraper for Amazon prices"
• "Build me a REST API with FastAPI and SQLite"
• "Debug this code" _(then send your file)_
• "What does this image show?" _(send an image)_
• "Write a cold email for a freelance pitch"

*For a full project:*
`/build a crypto trading bot with RSI strategy and Binance API`
`/build an Instagram automation tool with scheduling`
`/build a SaaS landing page with pricing and contact form`

*Voice:*
Send a voice note — I'll transcribe and reply.
Use /voice to also hear my replies as audio.

*Files & Images:*
Send any `.py .js .html .txt .csv .json` file and ask questions about it.
Send a photo/screenshot and I'll analyse it.
    """, parse_mode="Markdown")


@private_only
async def voice_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    new_state = await db.toggle_voice(user_id)
    status = "🔊 *Voice replies ON* — I'll speak my responses." if new_state \
             else "🔇 *Voice replies OFF* — Text only."
    await update.message.reply_text(status, parse_mode="Markdown")


@private_only
async def clear_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🗑️ Conversation memory cleared! Starting fresh."
    )


@private_only
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    stats = await db.get_user_stats(user_id)
    voice_on = await db.get_voice_enabled(user_id)

    await update.message.reply_text(f"""
📊 *Your Statistics*

💬 *Total messages:* {stats['total_messages']}
🗓️ *Today:* {stats['conversations_today']} conversations
📁 *Active projects:* {stats['active_projects']}
✅ *Completed projects:* {stats['completed_projects']}
🔊 *Voice replies:* {'ON' if voice_on else 'OFF'}
🕒 *Member since:* {stats['member_since'] or 'Unknown'}
    """, parse_mode="Markdown")


@private_only
async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show saved memories about the user."""
    user_id = update.effective_user.id
    memories = await db.get_memories(user_id)

    if not memories:
        await update.message.reply_text(
            "🧠 No memories saved yet. Just chat normally and I'll start "
            "remembering important things about you!"
        )
        return

    memory_list = "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories))
    await update.message.reply_text(
        f"🧠 *What I remember about you:*\n\n{memory_list}\n\n"
        f"Use /clearmemory to wipe all memories.",
        parse_mode="Markdown"
    )


@private_only
async def clear_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear all memories."""
    user_id = update.effective_user.id
    await db.clear_memories(user_id)
    await update.message.reply_text(
        "🧠 All memories cleared! I'll start fresh from now."
    )


@private_only
async def build_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger the project builder with /build <description>."""
    description = " ".join(context.args)

    if not description:
        await update.message.reply_text(
            "Please describe the project after the command.\n\n"
            "*Example:*\n`/build a crypto trading bot with RSI strategy`",
            parse_mode="Markdown"
        )
        return

    await _run_project_build(update, description)

# ─────────────────────────────────────────────
#  PROJECT BUILD RUNNER
# ─────────────────────────────────────────────

async def _run_project_build(update: Update, description: str):
    """Ask clarifying questions first, then build on next message."""
    user_id = update.effective_user.id

    # Ask clarifying questions
    await update.message.reply_text("🤔 Let me ask a few quick questions first...")
    questions = await builder.get_clarifying_questions(description)

    # Save description while waiting for answers
    pending_builds[user_id] = description

    await update.message.reply_text(questions)


async def _execute_build(update: Update, description: str, answers: str):
    """Execute the actual build with description + user answers."""
    user_id = update.effective_user.id

    # Combine description with answers for richer context
    full_spec = f"Project: {description}\n\nUser specifications:\n{answers}"

    project_id = await db.create_project(
        user_id=user_id,
        project_name=description[:60],
        project_type="generated",
        description=full_spec
    )

    async def progress(msg: str):
        await update.message.reply_text(msg)

    try:
        zip_path = await builder.build(full_spec, progress_callback=progress)

        with open(zip_path, "rb") as zf:
            await update.message.reply_document(
                document=zf,
                filename=os.path.basename(zip_path),
                caption="📦 Your project is ready! Unzip and follow the README."
            )

        await db.update_project_progress(project_id, 100, "completed")

        if os.path.exists(zip_path):
            os.remove(zip_path)

    except Exception as e:
        logger.error(f"Build error: {e}")
        await update.message.reply_text(
            f"❌ Build failed: {str(e)}\nPlease try again."
        )

# ─────────────────────────────────────────────
#  MESSAGE HANDLERS
# ─────────────────────────────────────────────

@private_only
@rate_limit
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages."""
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.first_name or "User"

    await db.add_or_update_user(
        user_id, update.effective_user.username or "",
        update.effective_user.first_name or "",
        update.effective_user.last_name
    )

    logger.info(f"[{username}] {user_message[:80]}")

    # Check if user is answering build questions
    if user_id in pending_builds:
        original_description = pending_builds.pop(user_id)
        await update.message.reply_text("🚀 Got it! Starting to build now...")
        await _execute_build(update, original_description, user_message)
        return

    # Check if message contains a URL
    import re
    url_pattern = re.compile(r'https?://[^\s]+')
    urls = url_pattern.findall(user_message)
    if urls:
        await update.message.reply_text("🌐 Reading that URL...")
        url_content = await read_url(urls[0])
        user_message = f"{user_message}\n\nURL content:\n{url_content}"

    # Natural language build trigger
    lower = user_message.lower()
    if any(lower.startswith(t) for t in BUILD_TRIGGERS):
        await _run_project_build(update, user_message)
        return

    # Web search trigger
    search_context = ""
    if tavily and any(t in lower for t in SEARCH_TRIGGERS):
        await update.message.reply_text("🔍 Searching the web...")
        search_results = await web_search(user_message)
        if search_results:
            await send_long_message(update, search_results)
            search_context = f"\n\nWeb search results to help answer:\n{search_results}"

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        ai_response = await get_ai_response(user_message + search_context, user_id)
        await send_long_message(update, ai_response)

        if await db.get_voice_enabled(user_id):
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.RECORD_VOICE
            )
            await send_voice_reply(update, ai_response)

    except Exception as e:
        logger.error(f"Message error: {e}")
        await update.message.reply_text(
            "Sorry, something went wrong. Please try again."
        )


@private_only
@rate_limit
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Transcribe voice messages and respond."""
    user_id = update.effective_user.id
    username = update.effective_user.first_name or "User"
    ogg_path = tempfile.mktemp(suffix=".ogg")

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(ogg_path)

        transcription = await transcribe_voice(ogg_path)
        logger.info(f"[{username}] Voice: {transcription[:80]}")

        await update.message.reply_text(
            f"🎤 *I heard:* _{transcription}_",
            parse_mode="Markdown"
        )

        if not transcription.strip():
            await update.message.reply_text(
                "I couldn't make out what you said. Please try again."
            )
            return

        ai_response = await get_ai_response(transcription, user_id)
        await send_long_message(update, ai_response)

        if await db.get_voice_enabled(user_id):
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.RECORD_VOICE
            )
            await send_voice_reply(update, ai_response)

    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't process your voice message. Please try again."
        )
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)


@private_only
@rate_limit
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Read and analyse uploaded text files."""
    user_id = update.effective_user.id
    document = update.message.document
    filename = document.file_name or "file"
    ext = Path(filename).suffix.lower()
    caption = update.message.caption or f"Analyse this file: {filename}"

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    if ext not in TEXT_EXTENSIONS:
        await update.message.reply_text(
            f"📄 I received `{filename}` but I can only read text-based files.\n\n"
            f"Supported: `{' '.join(sorted(TEXT_EXTENSIONS))}`",
            parse_mode="Markdown"
        )
        return

    tmp_path = tempfile.mktemp(suffix=ext)

    try:
        file = await context.bot.get_file(document.file_id)
        await file.download_to_drive(tmp_path)

        content = Path(tmp_path).read_text(encoding="utf-8", errors="replace")

        if len(content) > 12000:
            content = content[:12000] + "\n\n... [file truncated for length]"

        prompt = f"{caption}\n\n```{ext.lstrip('.')}\n{content}\n```"
        ai_response = await get_ai_response(prompt, user_id)

        await update.message.reply_text(ai_response, parse_mode="Markdown")

        if await db.get_voice_enabled(user_id):
            await send_voice_reply(update, ai_response)

    except Exception as e:
        logger.error(f"Document error: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't read that file. Please try again."
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@private_only
@rate_limit
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyse images using Claude vision."""
    user_id = update.effective_user.id
    caption = update.message.caption or "What's in this image? Describe it in detail."
    tmp_path = tempfile.mktemp(suffix=".jpg")

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        await file.download_to_drive(tmp_path)

        with open(tmp_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

        extra_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            }
        ]

        ai_response = await get_ai_response(caption, user_id, extra_content)
        await send_long_message(update, ai_response)

        if await db.get_voice_enabled(user_id):
            await send_voice_reply(update, ai_response)

    except Exception as e:
        logger.error(f"Photo error: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't analyse that image. Please try again."
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@private_only
async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force a web search."""
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text(
            "Please provide a search query.\n*Example:* `/search latest AI news`",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text("🔍 Searching the web...")
    results = await web_search(query)
    if results:
        await send_long_message(update, results)
        # Get Claude to summarize
        summary = await get_ai_response(
            f"Summarize these search results concisely:\n{results}",
            update.effective_user.id
        )
        await send_long_message(update, summary)
    else:
        await update.message.reply_text("No results found. Try a different query.")


@private_only
async def projects_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all projects the user has built."""
    user_id = update.effective_user.id
    projects = await db.get_user_projects(user_id)

    if not projects:
        await update.message.reply_text(
            "📁 No projects yet! Use `/build <description>` to create one.",
            parse_mode="Markdown"
        )
        return

    text = "📁 *Your Projects:*\n\n"
    for i, p in enumerate(projects, 1):
        text += f"{i}. *{p.name}*\n"
        text += f"   Type: {p.project_type}\n"
        text += f"   Phase: {p.current_phase}\n"
        text += f"   Progress: {p.completion_percentage}%\n"
        text += f"   Created: {p.created_at}\n\n"

    await send_long_message(update, text)


@private_only
async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Summarize the current conversation."""
    user_id = update.effective_user.id
    history = await db.get_conversation_history(user_id, limit=20)

    if not history:
        await update.message.reply_text(
            "No conversation history to summarize yet!"
        )
        return

    await update.message.reply_text("📝 Summarizing our conversation...")

    convo_text = ""
    for msg in history:
        role = "You" if msg["role"] == "user" else "Bot"
        convo_text += f"{role}: {msg['content']}\n\n"

    summary = await get_ai_response(
        f"Summarize this conversation concisely, highlighting key topics, "
        f"decisions made, and any action items:\n\n{convo_text}",
        user_id
    )
    await send_long_message(update, f"📝 *Conversation Summary:*\n\n{summary}")


@private_only
async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set a reminder — stores it as a high importance memory."""
    user_id = update.effective_user.id
    reminder = " ".join(context.args)

    if not reminder:
        await update.message.reply_text(
            "Please provide a reminder.\n*Example:* `/remind check trading bot performance tomorrow`",
            parse_mode="Markdown"
        )
        return

    await db.save_memory(
        user_id,
        f"REMINDER: {reminder}",
        category="reminder",
        importance=3
    )

    await update.message.reply_text(
        f"⏰ Reminder saved!\n\n*{reminder}*\n\n"
        f"I'll keep this in mind in our conversations. "
        f"Use `/memory` to see all your reminders.",
        parse_mode="Markdown"
    )


@private_only
async def imagine_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an image using Flux."""
    prompt = " ".join(context.args)

    if not prompt:
        await update.message.reply_text(
            "Please describe the image you want.\n\n"
            "*Example:*\n`/imagine a futuristic city at sunset, cyberpunk style`",
            parse_mode="Markdown"
        )
        return

    if not REPLICATE_API_KEY:
        await update.message.reply_text(
            "Image generation is not configured. Add REPLICATE_API_KEY to Railway Variables."
        )
        return

    await update.message.reply_text(f"🎨 Generating: _{prompt}_...", parse_mode="Markdown")
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
    )

    image_data = await generate_image(prompt)

    if image_data:
        await update.message.reply_photo(
            photo=image_data,
            caption=f"🎨 {prompt}"
        )
    else:
        await update.message.reply_text(
            "Sorry, image generation failed. Please try again with a different prompt."
        )


@private_only
async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get current weather for a city."""
    city = " ".join(context.args) or "Irvine"
    await update.message.reply_text(f"🌤️ Getting weather for {city}...")
    weather = await get_weather(city)
    await send_long_message(update, weather)


@private_only
async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a daily briefing."""
    user_id = update.effective_user.id
    await update.message.reply_text("📰 Generating your daily briefing...")
    briefing = await generate_briefing(user_id)
    await send_long_message(update, f"☀️ *Your Daily Briefing*\n\n{briefing}")

    if await db.get_voice_enabled(user_id):
        await send_voice_reply(update, briefing)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler."""
    logger.error("Unhandled exception:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "🔧 Something went wrong on my end. Please try again in a moment!"
        )

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

async def post_init(application):
    """Initialise the database after the app starts."""
    await db.init_database()
    logger.info("✅ Database initialised")


def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set")
        return
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        return

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    application.add_handler(CommandHandler("start",       start))
    application.add_handler(CommandHandler("help",        help_command))
    application.add_handler(CommandHandler("voice",       voice_toggle))
    application.add_handler(CommandHandler("build",       build_command))
    application.add_handler(CommandHandler("clear",       clear_context))
    application.add_handler(CommandHandler("stats",       stats_command))
    application.add_handler(CommandHandler("memory",      memory_command))
    application.add_handler(CommandHandler("clearmemory", clear_memory_command))
    application.add_handler(CommandHandler("search",      search_command))
    application.add_handler(CommandHandler("projects",    projects_command))
    application.add_handler(CommandHandler("summarize",   summarize_command))
    application.add_handler(CommandHandler("remind",      remind_command))
    application.add_handler(CommandHandler("weather",     weather_command))
    application.add_handler(CommandHandler("briefing",    briefing_command))
    application.add_handler(CommandHandler("imagine",     imagine_command))

    # Messages
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(filters.VOICE,        handle_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO,        handle_photo))

    # Errors
    application.add_error_handler(error_handler)

    logger.info("🤖 Bot starting...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
