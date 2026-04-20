import os
import json
import re
import zipfile
import tempfile
import logging
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────

CLARIFY_PROMPT = """You are a senior software architect taking a project brief.
The user wants to build something. Ask 3-5 SHORT, specific clarifying questions
to understand exactly what they need before building.

Focus on:
- Tech stack / language preference
- Specific APIs or services needed (and if they have keys)
- Key features they definitely want
- Any constraints (budget, hosting, complexity)
- Target platform (web, mobile, desktop, CLI)

Format your response EXACTLY like this:
Before I build this, I have a few quick questions:

1. [question]
2. [question]
3. [question]
4. [question]
5. [question]

Reply with your answers and I'll start building! 🚀"""

PLANNER_PROMPT = """You are an expert software architect.
The user will describe a project with specific requirements and answers to clarifying questions.
Your job is to plan the complete file structure.

Respond ONLY with a valid JSON array. No explanation, no markdown, no code fences.
Each item in the array must have:
  - "filename": relative path (e.g. "src/app.py")
  - "description": one sentence explaining what this file does

Keep descriptions short (under 120 characters) so the full plan fits comfortably.

Example output:
[
  {"filename": "main.py", "description": "Entry point of the application"},
  {"filename": "requirements.txt", "description": "Python dependencies"},
  {"filename": "README.md", "description": "Project documentation and setup guide"}
]"""

WRITER_PROMPT = """You are an expert software developer.
Write the complete, production-ready content for the file described below.
Use the exact specifications provided — specific APIs, libraries, and features requested.
Output ONLY the raw file content. No explanation, no markdown fences, no comments
about what you're doing — just the file content itself, ready to save and run.

Important:
- Use real, working code based on the specifications
- Add clear comments explaining key sections
- Include proper error handling
- For API keys use environment variables with clear names
- Make the README include setup instructions and required env vars"""


# ─────────────────────────────────────────────
#  CORE BUILDER
# ─────────────────────────────────────────────

class ProjectBuilder:

    async def get_clarifying_quest