import os
import json
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

    async def get_clarifying_questions(self, description: str) -> str:
        """Generate smart clarifying questions based on the project description."""
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system=CLARIFY_PROMPT,
            messages=[{"role": "user", "content": description}]
        )
        return response.content[0].text

    async def plan_project(self, description: str) -> list:
        """Ask Claude to plan the file structure for a project."""
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=PLANNER_PROMPT,
            messages=[{"role": "user", "content": description}]
        )

        raw = response.content[0].text.strip()

        # Strip accidental markdown fences just in case
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Planner returned invalid JSON: {raw}")
            raise ValueError(f"Failed to parse project plan: {e}")

        return plan

    async def write_file(self, filename: str, description: str,
                          project_description: str) -> str:
        """Ask Claude to write the content of a single file."""
        prompt = (
            f"Project: {project_description}\n\n"
            f"File to write: {filename}\n"
            f"Purpose: {description}\n\n"
            f"Write the complete content for this file."
        )

        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            system=WRITER_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    async def build(self, description: str, progress_callback=None) -> str:
        """
        Full build pipeline.

        Args:
            description:       The user's project description.
            progress_callback: An async function(message: str) to send
                               status updates back to the user.

        Returns:
            Path to the generated .zip file.
        """

        async def update(msg: str):
            if progress_callback:
                await progress_callback(msg)
            logger.info(msg)

        # ── Step 1: Plan ──────────────────────────────────
        await update("🧠 Planning project structure...")
        plan = await self.plan_project(description)
        file_count = len(plan)
        await update(f"📋 Plan ready — {file_count} files to generate")

        # ── Step 2: Create temp workspace ─────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # ── Step 3: Write each file ───────────────────
            for i, item in enumerate(plan, 1):
                filename = item.get("filename", f"file_{i}.txt")
                file_desc = item.get("description", "")

                await update(f"✍️  [{i}/{file_count}] Writing `{filename}`...")

                try:
                    content = await self.write_file(
                        filename, file_desc, description
                    )
                except Exception as e:
                    logger.error(f"Failed to write {filename}: {e}")
                    content = f"# Error generating this file\n# {e}\n"

                # Create subdirectories if needed (e.g. src/app.py)
                file_path = tmp_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # ── Step 4: Zip everything ────────────────────
            await update("📦 Zipping project...")

            # Derive a clean project name from the description
            project_name = self._slugify(description)
            zip_path = tmp_path / f"{project_name}.zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in plan:
                    filename = item.get("filename", "")
                    file_path = tmp_path / filename
                    if file_path.exists():
                        zf.write(file_path, arcname=f"{project_name}/{filename}")

            # ── Step 5: Move zip out of tempdir ──────────
            # tempdir will be deleted — copy zip to /tmp first
            final_zip = Path(tempfile.gettempdir()) / f"{project_name}.zip"
            final_zip.write_bytes(zip_path.read_bytes())

        await update(f"✅ Done! Sending `{project_name}.zip`...")
        return str(final_zip)

    @staticmethod
    def _slugify(text: str, max_length: int = 40) -> str:
        """Convert a description into a clean filename-safe slug."""
        import re
        # Take first few words
        words = text.strip().lower().split()[:5]
        slug = "-".join(words)
        # Remove any character that isn't alphanumeric or a dash
        slug = re.sub(r"[^a-z0-9\-]", "", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug[:max_length] or "project"


# ─────────────────────────────────────────────
#  GLOBAL INSTANCE
# ─────────────────────────────────────────────
builder = ProjectBuilder()