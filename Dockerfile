FROM python:3.11-slim

# ffmpeg is required for voice message conversion (ogg ↔ mp3 ↔ opus)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all bot files
COPY claude_bot.py .
COPY database_manager.py .
COPY project_builder.py .

# Persistent storage for the database
RUN mkdir -p /app/data

# Temp workspace for project builds
RUN mkdir -p /app/workspace

CMD ["python", "claude_bot.py"]