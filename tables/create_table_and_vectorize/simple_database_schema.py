"""
Simple Database Schema - Clean Multi-Level Vector Storage

Simple, fast schema for:
- File level embeddings
- Class level embeddings
- Function level embeddings
- Basic metadata for exact matches

No over-engineering, just what works.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()
SCHEMA_NAME = os.getenv("PG_DB_SCHEMA")
class SimpleDocumentDatabase:
    """Simple database manager for document embeddings."""

    def __init__(self, database_config: Dict[str, Any]):
        self.config = database_config
        self.logger = logging.getLogger(__name__)
        self.pool = None

    async def connect(self):
        """Connect to PostgreSQL with pgvector."""
        try:
            self.pool = await asyncpg.create_pool(**self.config)
            self.logger.info("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            return False

    async def store_file_embedding(self, repo_id: str, file_path: str,
                                 embedding: List[float], enhanced_content: str,
                                 original_content: str, metadata: Dict[str, Any]):
        print("Store file embeddings")
        """Store file-level embedding and metadata."""
        if not self.pool:
            await self.connect()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            # Convert embedding list to vector string format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            # 🆕 Separate basic metadata from rich metadata
            basic_metadata = {
                'file_hash': metadata.get('file_hash', ''),
                'language': metadata.get('language', ''),
                'file_size': metadata.get('file_size', 0),
                'function_names': metadata.get('function_names', []),
                'class_names': metadata.get('class_names', []),
                'imports': metadata.get('imports', [])
            }

            # Store rich metadata (everything else)
            rich_metadata = {k: v for k, v in metadata.items()
                           if k not in ['file_hash', 'language', 'file_size', 'function_names', 'class_names', 'imports']}

            await conn.execute(f"""
                INSERT INTO {SCHEMA_NAME}.file_trail_embeddings
                (repo_id, file_path, file_hash, embedding, enhanced_content,
                 original_content, programming_language, file_size, function_names, class_names, imports, rich_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (repo_id, file_path)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    enhanced_content = EXCLUDED.enhanced_content,
                    rich_metadata = EXCLUDED.rich_metadata,
                    updated_at = NOW()
            """, repo_id, file_path, basic_metadata['file_hash'], embedding_str,
                enhanced_content, original_content, basic_metadata['language'],
                basic_metadata['file_size'], json.dumps(basic_metadata['function_names']),
                json.dumps(basic_metadata['class_names']), json.dumps(basic_metadata['imports']),
                json.dumps(rich_metadata))

    async def store_class_embedding(self, repo_id: str, file_path: str, class_name: str,
                                  embedding: List[float], enhanced_content: str,
                                  original_content: str, metadata: Dict[str, Any]):
        print("Store class embedding")
        """Store class-level embedding and metadata."""
        if not self.pool:
            await self.connect()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            # Convert embedding list to vector string format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            # 🆕 Separate basic metadata from rich metadata
            basic_metadata = {
                'parent_classes': metadata.get('base_classes', metadata.get('parent_classes', [])),
                'methods': metadata.get('methods', []),
                'properties': metadata.get('properties', []),
                'start_line': metadata.get('start_line'),
                'end_line': metadata.get('end_line')
            }

            # Store rich metadata (everything else)
            rich_metadata = {k: v for k, v in metadata.items()
                           if k not in ['base_classes', 'parent_classes', 'methods', 'properties', 'start_line', 'end_line']}

            await conn.execute(f"""
                INSERT INTO {SCHEMA_NAME}.class_trail_embeddings
                (repo_id, file_path, class_name, embedding, enhanced_content,
                 original_content, parent_classes, methods, properties, start_line, end_line, rich_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (repo_id, file_path, class_name)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    enhanced_content = EXCLUDED.enhanced_content,
                    rich_metadata = EXCLUDED.rich_metadata,
                    updated_at = NOW()
            """, repo_id, file_path, class_name, embedding_str, enhanced_content,
                original_content,
                json.dumps(basic_metadata.get('parent_classes', [])),
                json.dumps(basic_metadata.get('methods', [])),
                json.dumps(basic_metadata.get('properties', [])),
                basic_metadata.get('start_line'), basic_metadata.get('end_line'),
                json.dumps(rich_metadata))

    async def store_function_embedding(self, repo_id: str, file_path: str, function_name: str,
                                     embedding: List[float], enhanced_content: str,
                                     original_content: str, metadata: Dict[str, Any]):
        print("Store function embedding")
        """Store function-level embedding and metadata."""
        if not self.pool:
            await self.connect()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            # Convert embedding list to vector string format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            # 🆕 Separate basic metadata from rich metadata
            basic_metadata = {
                'parent_class': metadata.get('parent_class'),
                'parameters': metadata.get('parameters', []),
                'return_type': metadata.get('return_type', 'unknown'),
                'start_line': metadata.get('start_line'),
                'end_line': metadata.get('end_line')
            }

            # Store rich metadata (everything else)
            rich_metadata = {k: v for k, v in metadata.items()
                           if k not in ['parent_class', 'parameters', 'return_type', 'start_line', 'end_line']}

            await conn.execute(f"""
                INSERT INTO {SCHEMA_NAME}.function_trail_embeddings
                (repo_id, file_path, function_name, embedding, enhanced_content,
                 original_content, parent_class, parameters, return_type, start_line, end_line, rich_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (repo_id, file_path, function_name)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    enhanced_content = EXCLUDED.enhanced_content,
                    rich_metadata = EXCLUDED.rich_metadata,
                    updated_at = NOW()
            """, repo_id, file_path, function_name, embedding_str, enhanced_content,
                original_content, basic_metadata.get('parent_class'),
                json.dumps(basic_metadata.get('parameters', [])),
                basic_metadata.get('return_type'),
                basic_metadata.get('start_line'), basic_metadata.get('end_line'),
                json.dumps(rich_metadata))

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("✅ Database connections closed")


def create_simple_database(database_config: Dict[str, Any]) -> SimpleDocumentDatabase:
    """Factory function to create simple database instance."""
    return SimpleDocumentDatabase(database_config)