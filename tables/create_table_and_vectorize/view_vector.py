import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
SCHEMA_NAME = "agentic_poc"

async def view_vectors():

    conn = await asyncpg.connect(
    host=os.getenv('PG_DB_HOST'),
    port=os.getenv('PG_DB_PORT'),
    user=os.getenv('PG_DB_USER'),
    password=os.getenv('PG_DB_PASSWORD'),
    database=os.getenv('PG_DB_NAME')  # Connect to default postgres database first
    )

    rows = await conn.fetch(f"SELECT * FROM {SCHEMA_NAME}.function_trail_embeddings")

    print("Rows", rows)
    await conn.close()

if __name__ == "__main__":
    # Set up proper event loop for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(view_vectors())