# 🤖 Telegram Personal Assistant (Claude-powered)

A fully-featured Telegram bot powered by Claude AI — with **voice message support both ways**.

---

## ✨ Features

| Capability | Example |
|---|---|
| 🎤 Voice input | Send a voice message — bot transcribes and responds |
| 🔊 Voice output | Toggle /voice to hear replies as audio |
| 🌐 Build websites | "Make me a landing page for my bakery" |
| 🐍 Write code | "Python script to rename all JPEGs in a folder" |
| 📧 Draft emails | "Write a follow-up email after a job interview" |
| 📄 Analyse files | Send a .txt / .py / .html file and ask questions |
| 🗣️ Conversation memory | Remembers context across messages |
| 📎 Auto file download | Code blocks are sent as downloadable files |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- **ffmpeg** (required for voice messages)

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Step 1 — Create your Telegram Bot

1. Open Telegram → message **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **API token**

### Step 2 — Get your Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Create an API Key
3. Copy it

### Step 3 — Run the bot

```bash
cd telegram-assistant

# Install dependencies
pip3 install -r requirements.txt

# Set environment variables
export TELEGRAM_TOKEN=your_token_here
export ANTHROPIC_API_KEY=your_key_here
export ALLOWED_USER_ID=your_user_id_here   # optional

# Run
python3 bot.py
```

---

## 💬 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | Show examples |
| `/clear` | Reset conversation memory |
| `/voice` | Toggle voice replies on/off 🔊 |

---

## 🎤 Voice Messages

**Sending voice to the bot:**
Just hold the mic button in Telegram and speak. The bot will:
1. Transcribe your message (shows you what it heard)
2. Send it to Claude
3. Reply with both text AND a voice message

**Getting voice replies to text messages:**
Send `/voice` to toggle voice replies on. The bot will read its responses aloud.

---

## 🔧 Customisation

### Change the TTS voice
Set the `TTS_VOICE` environment variable. Some options:
- `en-US-GuyNeural` (default, American male)
- `en-US-JennyNeural` (American female)
- `en-GB-RyanNeural` (British male)
- `en-AU-NatashaNeural` (Australian female)
- `el-GR-AthinaNeural` (Greek female)
- `el-GR-NestorNeural` (Greek male)

Run `edge-tts --list-voices` to see all available voices.

### Change the personality
Edit the `SYSTEM_PROMPT` in `bot.py`.

---

## 🐳 Docker

```bash
docker build -t tg-assistant .
docker run -d \
  -e TELEGRAM_TOKEN=your_token \
  -e ANTHROPIC_API_KEY=your_key \
  -e ALLOWED_USER_ID=your_id \
  --name tg-assistant \
  tg-assistant
```

---

## 🛡️ Security

- Set `ALLOWED_USER_ID` so only you can use the bot
- Never commit your `.env` file or API keys
- Revoke and regenerate tokens if accidentally shared
