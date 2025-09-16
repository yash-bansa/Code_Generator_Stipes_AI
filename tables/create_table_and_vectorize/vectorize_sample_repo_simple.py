#!/usr/bin/env python3
"""
Simple Repository Vectorization Script - FAST & RELIABLE VERSION
This script provides a simplified alternative to the ultimate vectorization system.
Uses OpenAI embeddings for proper vector generation.

IMPORTANT: This script reads files directly from the local filesystem.
Git-related metadata (commit history, author info, etc.) will not be available.
Rich metadata extraction focuses on code structure and content analysis instead.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simple_document_generator import SimpleDocumentGenerator

# Load environment variables
load_dotenv()

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import simplified system



async def check_openai_api() -> bool:
    print("Inside check openai key")
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False

    print(f"✅ OpenAI API key found (ends with: ...{api_key[-4:]})")
    return True


async def check_repository_exists(repo_path: str) -> bool:
    print("Inside check repository exists")
    """Check if the sample repository exists."""
    path = Path(repo_path)
    if not path.exists():
        print(f"❌ ERROR: Repository path {repo_path} does not exist!")
        return False

    if not path.is_dir():
        print(f"❌ ERROR: {repo_path} is not a directory!")
        return False

    # Check for Python files
    python_files = list(path.rglob("*.py"))
    if not python_files:
        print(f"⚠️ WARNING: No Python files found in {repo_path}")
        return False

    print(f"✅ Repository found with {len(python_files)} Python files")
    return True


def get_database_config() -> Dict[str, Any]:
    print("inside get_database_config")
    """Get database configuration from environment or use defaults."""
    return {
        'host': os.getenv('PG_DB_HOST'),
        'port': os.getenv('PG_DB_PORT'),
        'user': os.getenv('PG_DB_USER'),
        'password': os.getenv('PG_DB_PASSWORD'),
        'database': os.getenv('PG_DB_NAME')
    }


async def setup_simple_generator() -> SimpleDocumentGenerator:
    """Set up the simple document generator."""
    print("🔧 Setting up simple document generator...")

    database_config = get_database_config()
    print(f"📊 Database: {database_config['user']}@{database_config['host']}:{database_config['port']}/{database_config['database']}")

    embedding_client = None
    llm_client = None

    try:
        print("inside embeddings clients try")
        if os.getenv("AZURE_OPENAI_API_KEY"):

            embedding_client = AzureOpenAIEmbeddings(
                model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION")
            )

            llm_client = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_CHAT_MODEL_DEPLOYMENT_NAME"),
                model=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                temperature=0.1
            )
            print("✅ OpenAI clients initialized")

        else:
            embedding_client = OpenAIEmbeddings(model="text-embedding-3-small")
            llm_client = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            print("✅ OpenAI clients initialized")

    except ImportError:
        print("⚠️  OpenAI clients not available - using dummy embeddings")
        embedding_client = None
        llm_client = None

    generator = SimpleDocumentGenerator(
        database_config,
        embedding_client=embedding_client,
        llm_client=llm_client
    )

    print("✅ Simple document generator initialized!")
    return generator


def show_metadata_info():
    """Show information about what metadata will be extracted."""
    print("\n📋 RICH METADATA EXTRACTION:")
    print("✅ Available from local files:")
    print("   • Function signatures & parameter types")
    print("   • Code complexity & quality indicators")
    print("   • Intent classification (auth, validation, API, etc.)")
    print("   • Error handling & logging patterns")
    print("   • Security & performance indicators")
    print("   • Documentation quality assessment")
    print("   • Architectural patterns (controllers, services, etc.)")
    print("")
    print("❌ Not available (requires git integration):")
    print("   • Commit history & author information")
    print("   • File modification dates from git")
    print("   • Branch & merge information")
    print("")
    print("💡 This provides excellent search quality focusing on code content!")


async def vectorize_repository_simple(generator: SimpleDocumentGenerator, repo_path: str) -> Dict[str, Any]:
    """Vectorize repository using simple sequential processing."""

    print(f"\n🚀 Starting SIMPLE vectorization of: {repo_path}")
    print("📝 This process will:")
    print("   • Process files with multi-level granularity (file/class/function)")
    print("   • Extract rich metadata from code structure")
    print("   • Generate LLM-enhanced content for better embeddings")
    print("   • Create real OpenAI embeddings (not dummy vectors)")
    print("   • Store in clean 3-table schema with rich metadata")
    print("\n⏱️  Expected time: 2-5 minutes...")

    start_time = time.time()

    # Create a unique repository ID
    repo_id = f"simple_repo_sample_repo"  # 🆕 Use consistent repo_id instead of timestamp

    try:
        # Get list of Python files
        python_files = list(Path(repo_path).rglob("*.py"))
        total_files = len(python_files)
        processed_files = 0

        print(f"\n📁 Found {total_files} Python files to process")
        print("🔍 Extracting rich metadata from each code block...")

        # Index entire repository at once
        result = await generator.index_repository(repo_path, repo_id)

        if result["success"]:
            processed_files = result["files_processed"]
            blocks_created = result.get("total_blocks", 0)
            print(f"✅ Successfully indexed {processed_files} files")
            print(f"📊 Created {blocks_created} code blocks with rich metadata")
        else:
            print(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
            processed_files = 0

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n🎯 SIMPLE VECTORIZATION COMPLETED!")
        print(f"⏱️  Total time: {processing_time:.2f} seconds")
        print(f"📁 Files processed: {processed_files}/{total_files}")
        print(f"📊 Success rate: {(processed_files/total_files)*100:.1f}%")

        if processed_files > 0:
            print(f"\n💾 Rich metadata stored for:")
            print(f"   • Function signatures & types")
            print(f"   • Code quality indicators")
            print(f"   • Intent classifications")
            print(f"   • Architecture patterns")
            print(f"   • Security & performance markers")

        return {
            'status': 'simple_success',
            'files_processed': processed_files,
            'total_files': total_files,
            'processing_time': processing_time,
            'repo_id': repo_id,
            'blocks_created': result.get("total_blocks", 0)
        }

    except Exception as e:
        print(f"❌ ERROR during simple vectorization: {e}")
        print("💡 Note: Rich metadata extraction gracefully handles missing git data")
        raise


async def main():
    """Main execution function."""
    print("🚀 SIMPLE REPOSITORY VECTORIZATION")
    print("=" * 50)
    print("This script provides clean, fast vectorization with:")
    print("• Multi-level embeddings (file/class/function)")
    print("• Rich metadata extraction from code structure")
    print("• Real OpenAI embeddings (not dummy vectors)")
    print("• LLM-enhanced content for better search")
    print("• Clean 3-table database schema")
    print("• Enhanced LLM reranker with metadata")
    print("• Better error handling")
    print("")
    print("⚠️  Note: Reading from local files (no git metadata)")
    print("=" * 50)

    # Check prerequisites
    print("\n🔍 Checking prerequisites...")

    if not await check_openai_api():
        return

    repo_path = "<enter_the_repo_path_here>"
    if not await check_repository_exists(repo_path):
        return

    # Show metadata information
    show_metadata_info()

    try:
        # Set up simple generator
        generator = await setup_simple_generator()

        # Initialize database schema
        print("\n🗄️  Initializing database...")
        # initialized = await generator.initialize()
        # if not initialized:
        #     print("❌ Failed to initialize database")
        #     return 1

        # Vectorize the repository
        result = await vectorize_repository_simple(generator, repo_path)

        print(f"\n🎉 SIMPLE VECTORIZATION COMPLETE!")
        print(f"Repository ID: {result['repo_id']}")
        print(f"Files processed: {result['files_processed']}/{result['total_files']}")
        print(f"Code blocks created: {result['blocks_created']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")

        print(f"\n💡 Next steps:")
        print(f"• Use the repository ID '{result['repo_id']}' for searches")
        print(f"• Rich metadata enables intelligent LLM reranking")
        print(f"• Search quality should be excellent despite missing git data")
        print(f"• Try complex queries like 'authentication functions' or 'error handling'")

        if result['blocks_created'] > 0:
            print(f"\n🔍 Search capabilities enhanced with:")
            print(f"   • Function signature matching")
            print(f"   • Intent-based filtering")
            print(f"   • Quality-based ranking")
            print(f"   • Architecture-aware results")

    except KeyboardInterrupt:
        print(f"\n⏹️  Vectorization interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Rich metadata extraction gracefully handles missing git data")
        print(f"   This simple version should provide excellent search quality")
        return 1

    return 0


if __name__ == "__main__":
    # Set up proper event loop for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)