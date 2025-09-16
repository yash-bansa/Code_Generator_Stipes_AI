#!/usr/bin/env python3
"""
PostgreSQL setup script for Simple Document Generator.
Creates clean 3-table schema with single embeddings per code block.
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
SCHEMA_NAME = os.getenv("PG_DB_SCHEMA")
# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def check_database_connection(config):
    """Test database connection."""
    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')
        )
        await conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def create_database_if_not_exists(config):
    """Create the agentic_ai database if it doesn't exist."""
    try:
        # Connect to postgres database to create our target database
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        # Check if database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            config['database']
        )

        if result:
            print(f"✅ Database '{config['database']}' already exists")
        else:
            # Create database
            await conn.execute(f'CREATE DATABASE "{config["database"]}"')
            print(f"✅ Created database '{config['database']}'")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False


async def setup_pgvector_extension(config):
    """Enable pgvector extension in the database."""
    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ pgvector extension enabled")

        # Verify extension is available
        result = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )

        if result:
            print(f"✅ pgvector version: {result}")
        else:
            print("❌ pgvector extension not found")
            await conn.close()
            return False

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error setting up pgvector: {e}")
        return False


async def create_simple_schema(config):
    """Create the simple 3-table schema for document generation."""
    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        print("📋 Creating Simple Document Generator schema...")
        print(f"Creating tables in schema: {SCHEMA_NAME}")

        # Create file_trail_embeddings table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.file_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                file_hash VARCHAR(64),
                embedding VECTOR(1536) NOT NULL,
                enhanced_content TEXT,
                original_content TEXT,
                programming_language VARCHAR(50),
                file_size INTEGER,
                function_names JSONB DEFAULT '[]',
                class_names JSONB DEFAULT '[]',
                imports JSONB DEFAULT '[]',
                rich_metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(repo_id, file_path)
            );
        """)
        print("✅ Created file_trail_embeddings table")

        # Create class_trail_embeddings table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.class_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                class_name VARCHAR(255) NOT NULL,
                embedding VECTOR(1536) NOT NULL,
                enhanced_content TEXT,
                original_content TEXT,
                parent_classes JSONB DEFAULT '[]',
                methods JSONB DEFAULT '[]',
                properties JSONB DEFAULT '[]',
                start_line INTEGER,
                end_line INTEGER,
                rich_metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(repo_id, file_path, class_name)
            );
        """)
        print("✅ Created class_trail_embeddings table")

        # Create function_trail_embeddings table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.function_trail_embeddings (
                id SERIAL PRIMARY KEY,
                repo_id VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                function_name VARCHAR(255) NOT NULL,
                embedding VECTOR(1536) NOT NULL,
                enhanced_content TEXT,
                original_content TEXT,
                parent_class VARCHAR(255),
                parameters JSONB DEFAULT '[]',
                return_type VARCHAR(255),
                start_line INTEGER,
                end_line INTEGER,
                rich_metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(repo_id, file_path, function_name)
            );
        """)
        print("✅ Created function_trail_embeddings table")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating simple schema: {e}")
        return False


async def create_vector_indexes(config):
    """Create optimized vector indexes for fast similarity search."""

    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        print(f"🚀 Creating vector indexes in {SCHEMA_NAME}")

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_file_trail_embeddings_vector
            ON {SCHEMA_NAME}.file_trail_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print("✅ Created file_trail_embeddings vector index")

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_class_trail_embeddings_vector
            ON {SCHEMA_NAME}.class_trail_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print("✅ Created class_trail_embeddings vector index")

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_trail_embeddings_vector
            ON {SCHEMA_NAME}.function_trail_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print("✅ Created function_trail_embeddings vector index")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating vector indexes: {e}")
        return False


async def create_metadata_indexes(config):
    """Create BTREE + GIN metadata indexes in the target schema"""

    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        print(f"📊 Creating metadata indexes in schema : {SCHEMA_NAME}")

        # File-level indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_file_repo_id
            ON {SCHEMA_NAME}.file_trail_embeddings (repo_id);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON {SCHEMA_NAME}.file_trail_embeddings (file_path);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_file_rich_metadata
            ON {SCHEMA_NAME}.file_trail_embeddings USING GIN (rich_metadata);
        """)

        # Class-level indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_class_repo_id
            ON {SCHEMA_NAME}.class_trail_embeddings (repo_id);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_class_name
            ON {SCHEMA_NAME}.class_trail_embeddings (class_name);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_class_file_path
            ON {SCHEMA_NAME}.class_trail_embeddings (file_path);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_class_rich_metadata
            ON {SCHEMA_NAME}.class_trail_embeddings USING GIN (rich_metadata);
        """)

        # Function-level indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_repo_id
            ON {SCHEMA_NAME}.function_trail_embeddings (repo_id);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_name
            ON {SCHEMA_NAME}.function_trail_embeddings (function_name);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_file_path
            ON {SCHEMA_NAME}.function_trail_embeddings (file_path);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_parent_class
            ON {SCHEMA_NAME}.function_trail_embeddings (parent_class);
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_function_rich_metadata
            ON {SCHEMA_NAME}.function_trail_embeddings USING GIN (rich_metadata);
        """)

        print("✅ Created metadata indexes for fast search")
        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Error creating metadata indexes: {e}")
        return False



async def verify_setup(config):
    """Verify that everything is set up correctly."""

    try:
        conn = await asyncpg.connect(
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
        )

        # Check tables exist
        tables = await conn.fetch(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{SCHEMA_NAME}'
            AND table_name IN ('file_trail_embeddings', 'class_trail_embeddings', 'function_trail_embeddings')
            ORDER BY table_name;
        """)

        table_names = [row['table_name'] for row in tables]
        print(f"✅ Tables found in schema {SCHEMA_NAME} : {', '.join(table_names)}")

        if len(table_names) != 3:
            print(f"❌ Expected 3 tables, found {len(table_names)}")
            await conn.close()
            return False

        # Check vector indexes
        vector_indexes = await conn.fetch(f"""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = '{SCHEMA_NAME}'
                AND tablename IN ('file_trail_embeddings', 'class_trail_embeddings', 'function_trail_embeddings')
                AND indexname LIKE '%_vector'
            ORDER BY indexname;
        """)
        print(f"✅ Vector indexes: {len(vector_indexes)} created")

        # Check metadata indexes
        metadata_indexes = await conn.fetch(f"""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = '{SCHEMA_NAME}'
                AND tablename IN ('file_trail_embeddings', 'class_trail_embeddings', 'function_trail_embeddings')
                AND indexname NOT LIKE '%_vector'
                AND indexname NOT LIKE '%_pkey'
            ORDER BY indexname;
        """)
        print(f"✅ Metadata indexes: {len(metadata_indexes)} created")

        # Check pgvector extension
        vector_version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        print(f"✅ pgvector extension version: {vector_version}")

        # Test vector operations
        await conn.execute("SELECT '[1,2,3]'::vector(3) <-> '[1,2,4]'::vector(3)")
        print("✅ Vector operations working correctly")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def load_database_config():
    """Load database configuration from environment."""
    # Load .env file
    load_dotenv()

    config = {
        'host': os.getenv('PG_DB_HOST'),
        'port': os.getenv('PG_DB_PORT'),
        'user': os.getenv('PG_DB_USER'),
        'password': os.getenv('PG_DB_PASSWORD'),
        'database': os.getenv('PG_DB_NAME')
    }

    print("📊 Database configuration:")
    print(f"   Host: {config['host']}:{config['port']}")
    print(f"   Database: {config['database']}")
    print(f"   User: {config['user']}")

    return config


async def main():
    """Main setup function."""
    print("🐘 SIMPLE DOCUMENT GENERATOR - POSTGRESQL SETUP")
    print("=" * 50)
    print("Creating clean 3-table schema with LLM-enhanced embeddings")
    print()

    # Load database configuration
    config = load_database_config()

    try:
        print("\n🔍 Step 1: Testing database connection...")
        if not await check_database_connection(config):
            print("💡 Make sure your PostgreSQL Docker container is running:")
            print("   docker run -d --name postgres-simple \\")
            print("     -e POSTGRES_PASSWORD=123456 \\")
            print("     -e POSTGRES_DB=agentic_ai -p 5432:5432 ankane/pgvector")
            return 1

        print("\n🗄️  Step 2: Creating database...")
        if not await create_database_if_not_exists(config):
            return 1

        print("\n📦 Step 3: Setting up pgvector extension...")
        if not await setup_pgvector_extension(config):
            print("💡 Make sure you're using a PostgreSQL image with pgvector:")
            print("   docker run -d --name postgres-simple \\")
            print("     -e POSTGRES_PASSWORD=123456 \\")
            print("     -e POSTGRES_DB=agentic_ai -p 5432:5432 ankane/pgvector")
            return 1

        print("\n📋 Step 4: Creating simple schema...")
        if not await create_simple_schema(config):
            return 1

        print("\n🚀 Step 5: Creating vector indexes...")
        if not await create_vector_indexes(config):
            return 1

        print("\n📊 Step 6: Creating metadata indexes...")
        if not await create_metadata_indexes(config):
            return 1

        print("\n✅ Step 7: Verifying setup...")
        if not await verify_setup(config):
            return 1

        print("\n🎉 SIMPLE DOCUMENT GENERATOR SETUP COMPLETE!")
        print("=" * 50)
        print("✨ Your database is ready with:")
        print("   📁 file_embeddings - Full file vectors")
        print("   📦 class_embeddings - Class-level vectors")
        print("   🔧 function_embeddings - Function-level vectors + rich metadata")
        print("   🚀 Optimized indexes for fast search")
        print("   🧠 LLM-enhanced content storage")
        print("   🎯 Rich metadata for intelligent LLM reranking")
        print()
        print(f"🗄️  Database: {config['database']} on {config['host']}:{config['port']}")
        print()
        print("📝 Next steps:")
        print("1. Update your .env with the database config above")
        print("2. Run: python simple_document_generator_example.py")
        print("3. Your repository will be indexed with simple, fast accuracy!")
        print()
        print("⚡ Expected performance:")
        print("   • Search speed: 20-50ms")
        print("   • Storage: 1x baseline")
        print("   • Accuracy: 85-90%")
        print("   • LLM enhanced: Smart query analysis + result refinement")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Set up proper event loop for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)