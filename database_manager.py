import aiosqlite
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    id: int
    name: str
    project_type: str
    current_phase: str
    completion_percentage: int
    technologies: Dict
    created_at: datetime
    last_activity: datetime


@dataclass
class ConversationContext:
    user_id: int
    project_id: Optional[int]
    messages: List[Dict]
    preferences: Dict


class DatabaseManager:
    def __init__(self, db_path="bot.db"):
        self.db_path = db_path

    async def init_database(self):
        """Initialize all database tables"""
        async with aiosqlite.connect(self.db_path) as db:

            # Users table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    voice_enabled BOOLEAN DEFAULT FALSE,
                    is_premium BOOLEAN DEFAULT FALSE
                )
            """)

            # Projects table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    project_name TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    description TEXT,
                    current_phase TEXT DEFAULT 'planning',
                    completion_percentage INTEGER DEFAULT 0,
                    technologies JSON,
                    specifications JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Conversations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    project_id INTEGER,
                    message TEXT,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_type TEXT DEFAULT 'chat',
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # Project tasks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS project_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    task_name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'medium',
                    estimated_hours INTEGER,
                    actual_hours INTEGER,
                    phase TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # Project files table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS project_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    filename TEXT NOT NULL,
                    file_type TEXT,
                    purpose TEXT,
                    content TEXT,
                    status TEXT DEFAULT 'created',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # User preferences table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    coding_style JSON,
                    preferred_technologies JSON,
                    communication_style TEXT DEFAULT 'professional',
                    notification_settings JSON,
                    project_preferences JSON,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Project decisions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS project_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    decision_type TEXT,
                    decision TEXT,
                    reasoning TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            await db.commit()
            logger.info("Database initialized successfully")

    async def add_or_update_user(self, user_id: int, username: str,
                                  first_name: str, last_name: str = None):
        """Add or update user information"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO users (user_id, username, first_name, last_name, last_activity, total_messages)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    last_activity = excluded.last_activity,
                    total_messages = total_messages + 1
            """, (user_id, username, first_name, last_name, datetime.now()))
            await db.commit()

    async def get_voice_enabled(self, user_id: int) -> bool:
        """Check if voice replies are enabled for a user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT voice_enabled FROM users WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            return bool(row[0]) if row else False

    async def toggle_voice(self, user_id: int) -> bool:
        """Toggle voice replies for a user, returns new state"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE users SET voice_enabled = NOT voice_enabled WHERE user_id = ?
            """, (user_id,))
            await db.commit()
        return await self.get_voice_enabled(user_id)

    async def create_project(self, user_id: int, project_name: str,
                              project_type: str, description: str = None,
                              specifications: Dict = None) -> int:
        """Create a new project"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO projects (user_id, project_name, project_type, description, specifications)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, project_name, project_type, description,
                  json.dumps(specifications) if specifications else None))

            project_id = cursor.lastrowid
            await db.commit()

            logger.info(f"Created project '{project_name}' with ID {project_id}")
            return project_id

    async def get_user_projects(self, user_id: int,
                                 status: str = 'active') -> List[ProjectInfo]:
        """Get all projects for a user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, project_name, project_type, current_phase,
                       completion_percentage, technologies, created_at, last_activity
                FROM projects
                WHERE user_id = ? AND status = ?
                ORDER BY last_activity DESC
            """, (user_id, status))

            rows = await cursor.fetchall()
            projects = []
            for row in rows:
                projects.append(ProjectInfo(
                    id=row[0],
                    name=row[1],
                    project_type=row[2],
                    current_phase=row[3],
                    completion_percentage=row[4],
                    technologies=json.loads(row[5]) if row[5] else {},
                    created_at=row[6],
                    last_activity=row[7]
                ))
            return projects

    async def get_project_context(self, project_id: int) -> Dict:
        """Get complete project context including conversations, tasks, files"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            )
            project = await cursor.fetchone()

            if not project:
                return None

            # Recent conversations (last 20, oldest first)
            cursor = await db.execute("""
                SELECT message, response, timestamp, message_type
                FROM conversations
                WHERE project_id = ?
                ORDER BY timestamp DESC LIMIT 20
            """, (project_id,))
            conversations = list(reversed(await cursor.fetchall()))

            # Tasks
            cursor = await db.execute("""
                SELECT task_name, description, status, priority,
                       estimated_hours, actual_hours, phase
                FROM project_tasks
                WHERE project_id = ?
                ORDER BY created_at
            """, (project_id,))
            tasks = await cursor.fetchall()

            # Files
            cursor = await db.execute("""
                SELECT filename, file_type, purpose, status, created_at
                FROM project_files
                WHERE project_id = ?
                ORDER BY created_at
            """, (project_id,))
            files = await cursor.fetchall()

            # Decisions
            cursor = await db.execute("""
                SELECT decision_type, decision, reasoning, timestamp
                FROM project_decisions
                WHERE project_id = ?
                ORDER BY timestamp
            """, (project_id,))
            decisions = await cursor.fetchall()

            return {
                'project': project,
                'conversations': conversations,
                'tasks': tasks,
                'files': files,
                'decisions': decisions
            }

    async def log_conversation(self, user_id: int, message: str, response: str,
                                project_id: int = None, message_type: str = 'chat'):
        """Log a conversation to the database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations (user_id, project_id, message, response, message_type)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, project_id, message, response, message_type))
            await db.commit()

    async def get_conversation_history(self, user_id: int,
                                        limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a user as Claude-ready messages"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT message, response FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit))
            rows = await cursor.fetchall()

        messages = []
        for message, response in reversed(rows):
            messages.append({"role": "user", "content": message})
            messages.append({"role": "assistant", "content": response})
        return messages

    async def update_project_progress(self, project_id: int,
                                       completion_percentage: int,
                                       current_phase: str = None):
        """Update project progress"""
        async with aiosqlite.connect(self.db_path) as db:
            if current_phase:
                await db.execute("""
                    UPDATE projects
                    SET completion_percentage = ?, current_phase = ?, last_activity = ?
                    WHERE id = ?
                """, (completion_percentage, current_phase, datetime.now(), project_id))
            else:
                await db.execute("""
                    UPDATE projects
                    SET completion_percentage = ?, last_activity = ?
                    WHERE id = ?
                """, (completion_percentage, datetime.now(), project_id))
            await db.commit()

    async def add_project_task(self, project_id: int, task_name: str,
                                description: str = None, estimated_hours: int = None,
                                phase: str = None, priority: str = 'medium'):
        """Add a task to a project"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO project_tasks
                (project_id, task_name, description, estimated_hours, phase, priority)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project_id, task_name, description, estimated_hours, phase, priority))
            await db.commit()

    async def add_project_file(self, project_id: int, filename: str,
                                file_type: str, purpose: str = None,
                                content: str = None):
        """Add a file record to a project"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO project_files
                (project_id, filename, file_type, purpose, content)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, filename, file_type, purpose, content))
            await db.commit()

    async def get_user_stats(self, user_id: int) -> Dict:
        """Get comprehensive user statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT total_messages, created_at, last_activity
                FROM users WHERE user_id = ?
            """, (user_id,))
            user_data = await cursor.fetchone()

            cursor = await db.execute("""
                SELECT status, COUNT(*) as count
                FROM projects WHERE user_id = ?
                GROUP BY status
            """, (user_id,))
            project_counts = {row[0]: row[1] async for row in cursor}

            cursor = await db.execute("""
                SELECT COUNT(*) FROM conversations
                WHERE user_id = ? AND date(timestamp) = date('now')
            """, (user_id,))
            conversations_today = (await cursor.fetchone())[0]

            return {
                'total_messages': user_data[0] if user_data else 0,
                'member_since': user_data[1] if user_data else None,
                'last_activity': user_data[2] if user_data else None,
                'active_projects': project_counts.get('active', 0),
                'completed_projects': project_counts.get('completed', 0),
                'conversations_today': conversations_today
            }

    async def find_project_by_context(self, user_id: int,
                                       message: str) -> Optional[int]:
        """Find an existing project based on message context"""
        message_lower = message.lower()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, project_name FROM projects
                WHERE user_id = ? AND status = 'active'
            """, (user_id,))

            async for project_id, project_name in cursor:
                if project_name.lower() in message_lower:
                    await db.execute("""
                        UPDATE projects SET last_activity = ? WHERE id = ?
                    """, (datetime.now(), project_id))
                    await db.commit()
                    return project_id

        return None


# Global instance
db = DatabaseManager()
