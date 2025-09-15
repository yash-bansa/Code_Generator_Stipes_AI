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

SCHEMA_NAME = "agentic_poc"
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

    async def create_schema(self) -> bool:
        """Create simple, clean database schema."""
        if not self.pool:
            await self.connect()

        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.logger.info("✅ Enabled pgvector extension")

                # Create simple tables
                await self._create_file_trail_embeddings_table(conn)
                await self._create_class_trail_embeddings_table(conn)
                await self._create_function_trail_embeddings_table(conn)
                await self._create_indexes(conn)

                self.logger.info("✅ Simple database schema created successfully")
                return True

        except Exception as e:
            self.logger.error(f"❌ Schema creation failed: {e}")
            return False

    async def _create_file_trail_embeddings_table(self, conn):
        """Create simple file-level embeddings table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS file_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                file_hash VARCHAR(64) NOT NULL,

                -- Single semantic embedding
                embedding VECTOR(1536) NOT NULL,

                -- Enhanced content (LLM-generated)
                enhanced_content TEXT,
                original_content TEXT,

                -- Basic metadata for exact searches
                programming_language VARCHAR(50),
                file_size INTEGER DEFAULT 0,
                function_names JSONB DEFAULT '[]',
                class_names JSONB DEFAULT '[]',
                imports JSONB DEFAULT '[]',

                -- Quality tracking
                enhancement_quality VARCHAR(20) DEFAULT 'good',

                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(repo_id, file_path)
            );
        """)
        self.logger.info("✅ Created file_embeddings table")

    async def _create_class_trail_embeddings_table(self, conn):
        """Create simple class-level embeddings table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS class_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                class_name VARCHAR(255) NOT NULL,

                -- Single semantic embedding
                embedding VECTOR(1536) NOT NULL,

                -- Enhanced content
                enhanced_content TEXT,
                original_content TEXT,

                -- Class metadata
                parent_classes JSONB DEFAULT '[]',
                methods JSONB DEFAULT '[]',
                properties JSONB DEFAULT '[]',

                -- Location in file
                start_line INTEGER,
                end_line INTEGER,

                -- 🆕 Rich metadata for better search
                rich_metadata JSONB DEFAULT '{}',

                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(repo_id, file_path, class_name)
            );
        """)
        self.logger.info("✅ Created class_embeddings table")

    async def _create_function_trail_embeddings_table(self, conn):
        """Create simple function-level embeddings table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS function_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                function_name VARCHAR(255) NOT NULL,

                -- Single semantic embedding
                embedding VECTOR(1536) NOT NULL,

                -- Enhanced content
                enhanced_content TEXT,
                original_content TEXT,

                -- Function metadata
                parent_class VARCHAR(255),
                parameters JSONB DEFAULT '[]',
                return_type VARCHAR(100),

                -- Location in file
                start_line INTEGER,
                end_line INTEGER,

                -- 🆕 Rich metadata for better search
                rich_metadata JSONB DEFAULT '{}',

                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(repo_id, file_path, function_name)
            );
        """)
        self.logger.info("✅ Created function_embeddings table")

    async def _create_indexes(self, conn):
        """Create essential indexes for fast search."""
        indexes = [
            # Vector similarity indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_embedding ON file_trail_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_class_embedding ON class_trail_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_function_embedding ON function_trail_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",

            # Metadata search indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_metadata ON file_trail_embeddings(repo_id, programming_language)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_function_names ON file_trail_embeddings USING GIN (function_names)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_class_names ON file_trail_embeddings USING GIN (class_names)",

            # Exact match indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_class_name ON class_trail_embeddings(repo_id, class_name)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_function_name ON function_trail_embeddings(repo_id, function_name)",

            # Rich metadata indexes (GIN for JSONB)
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_class_rich_metadata ON class_trail_embeddings USING GIN (rich_metadata)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_function_rich_metadata ON function_trail_embeddings USING GIN (rich_metadata)",
        ]

        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
                index_name = index_sql.split()[-1].split('(')[0] if '(' in index_sql else "index"
                self.logger.info(f"✅ Created {index_name}")
            except Exception as e:
                self.logger.warning(f"Index creation skipped (may exist): {e}")

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

            await conn.execute("""
                INSERT INTO file_trail_embeddings
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

            await conn.execute("""
                INSERT INTO class_trail_embeddings
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

            await conn.execute("""
                INSERT INTO function_trail_embeddings
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

    async def search_by_vector(self, query_embedding: List[float], repo_id: str,
                             search_level: str = "all", limit: int = 20):
        print("Inside search_by_vector in simple_database_schema")
        print("The search level specified here is", search_level)
        """Search using vector similarity."""
        if not self.pool:
            await self.connect()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            # Convert query embedding to vector string format
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            if search_level == "file" or search_level == "all":
                file_results = await conn.fetch("""
                    SELECT 'file' as type, file_path, file_path as name,
                           embedding <=> $1 as distance,
                           programming_language, enhanced_content, rich_metadata
                    FROM file_trail_embeddings
                    WHERE repo_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding_str, repo_id, limit)
            else:
                file_results = []

            if search_level == "class" or search_level == "all":
                class_results = await conn.fetch("""
                    SELECT 'class' as type, file_path, class_name as name,
                           embedding <=> $1 as distance,
                           enhanced_content, rich_metadata
                    FROM class_trail_embeddings
                    WHERE repo_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding_str, repo_id, limit)
            else:
                class_results = []

            if search_level == "function" or search_level == "all":
                function_results = await conn.fetch("""
                    SELECT 'function' as type, file_path, function_name as name,
                           embedding <=> $1 as distance,
                           enhanced_content, parent_class, rich_metadata
                    FROM function_trail_embeddings
                    WHERE repo_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding_str, repo_id, limit)
            else:
                function_results = []

            # Combine and sort all results
            all_results = list(file_results) + list(class_results) + list(function_results)
            self.logger.info(f"ALL Results: {all_results}")
            all_results.sort(key=lambda x: x['distance'])

            return all_results[:limit]

    async def search_by_metadata(self, repo_id: str, query_terms: List[str], limit: int = 10):
        """Fast metadata search for exact matches."""
        if not self.pool:
            await self.connect()
        if not self.pool:
            raise RuntimeError("Database connection failed")

        async with self.pool.acquire() as conn:
            results = []

            for term in query_terms:
                # Search function names
                function_matches = await conn.fetch("""
                    SELECT 'function' as type, file_path, function_name as name,
                           0.95 as score, enhanced_content, rich_metadata
                    FROM function_trail_embeddings
                    WHERE repo_id = $1 AND function_name ILIKE $2
                    LIMIT $3
                """, repo_id, f"%{term}%", limit)

                # Search class names
                class_matches = await conn.fetch("""
                    SELECT 'class' as type, file_path, class_name as name,
                           0.9 as score, enhanced_content, rich_metadata
                    FROM class_trail_embeddings
                    WHERE repo_id = $1 AND class_name ILIKE $2
                    LIMIT $3
                """, repo_id, f"%{term}%", limit)

                # Search file paths
                file_matches = await conn.fetch("""
                    SELECT 'file' as type, file_path, file_path as name,
                           0.85 as score, enhanced_content, rich_metadata
                    FROM file_trail_embeddings
                    WHERE repo_id = $1 AND file_path ILIKE $2
                    LIMIT $3
                """, repo_id, f"%{term}%", limit)

                results.extend(list(function_matches) + list(class_matches) + list(file_matches))

            # Remove duplicates and sort by score
            seen = set()
            unique_results = []
            for result in results:
                key = (result['type'], result['file_path'], result['name'])
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)

            unique_results.sort(key=lambda x: x['score'], reverse=True)
            return unique_results[:limit]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("✅ Database connections closed")


def create_simple_database(database_config: Dict[str, Any]) -> SimpleDocumentDatabase:
    """Factory function to create simple database instance."""
    return SimpleDocumentDatabase(database_config)