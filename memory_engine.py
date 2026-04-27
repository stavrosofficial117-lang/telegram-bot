import json
import logging
import os
from datetime import datetime
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────

MEMORY_EXTRACT_PROMPT = """You are a memory extraction system for a personal AI assistant.
Analyze the user's message and extract any important information worth remembering long-term.

Extract memories in these categories:
- preference: things they like/dislike, how they work
- skill: languages, tools, expertise they have
- project: ongoing work, goals, ideas
- personal: life context, location, schedule, relationships
- style: communication style, tone preferences
- goal: what they're trying to achieve long term

Respond ONLY with a JSON array. Each item must have:
- "memory": concise fact to remember (max 100 chars)
- "category": one of the categories above
- "importance": 1-3 (1=nice to know, 2=useful, 3=critical)

If nothing worth remembering, return: []

Examples:
User: "I hate JavaScript, always use Python"
[{"memory": "Strongly prefers Python, dislikes JavaScript", "category": "preference", "importance": 3}]

User: "what's 2+2"
[]

Respond ONLY with the JSON array."""


MEMORY_SUMMARIZE_PROMPT = """You are summarizing a user's memory profile for an AI assistant.
Given a list of memories, create a concise summary paragraph that captures:
- Who this person is
- What they work on
- Their key preferences and skills
- Any ongoing projects

Keep it under 200 words. Write in second person ("You are...", "You prefer...").
This summary will be injected into the AI's context."""


PROACTIVE_MEMORY_PROMPT = """You are analyzing a conversation to find relevant memories to surface.
Given the user's message and their memory list, identify which memories are most relevant
to the current conversation.

Return ONLY a JSON array of the most relevant memory strings (max 5).
If none are relevant, return [].

Respond ONLY with the JSON array."""


# ─────────────────────────────────────────────
#  MEMORY ENGINE
# ─────────────────────────────────────────────

class MemoryEngine:

    async def extract_memories(self, message: str) -> list[dict]:
        """Extract important memories from a user message."""
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=500,
                system=MEMORY_EXTRACT_PROMPT,
                messages=[{"role": "user", "content": message}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Memory extraction error: {e}")
            return []

    async def summarize_memories(self, memories: list[str]) -> str:
        """Summarize a list of memories into a coherent profile."""
        if not memories:
            return ""
        try:
            memory_text = "\n".join(f"- {m}" for m in memories)
            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                system=MEMORY_SUMMARIZE_PROMPT,
                messages=[{"role": "user", "content": memory_text}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Memory summarize error: {e}")
            return ""

    async def get_relevant_memories(self, message: str,
                                     memories: list[str]) -> list[str]:
        """Get the most relevant memories for the current message."""
        if not memories:
            return []
        try:
            content = f"User message: {message}\n\nAll memories:\n"
            content += "\n".join(f"- {m}" for m in memories)

            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                system=PROACTIVE_MEMORY_PROMPT,
                messages=[{"role": "user", "content": content}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Relevant memory error: {e}")
            return memories[:5]  # fallback to most recent

    def build_memory_context(self, summary: str,
                              relevant_memories: list[str],
                              stats: dict) -> str:
        """Build the memory context string to inject into Claude's system prompt."""
        context = "\n\n━━━ WHAT YOU KNOW ABOUT THIS USER ━━━\n"

        if summary:
            context += f"\n{summary}\n"

        if relevant_memories:
            context += "\nMost relevant to this conversation:\n"
            context += "\n".join(f"• {m}" for m in relevant_memories)

        if stats:
            context += f"\n\nStats: {stats.get('total_messages', 0)} total messages, "
            context += f"{stats.get('active_projects', 0)} active projects, "
            context += f"member since {stats.get('member_since', 'recently')}"

        context += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        return context


# Global instance
memory_engine = MemoryEngine()
