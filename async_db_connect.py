import os
import asyncpg
import asyncio
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

async def fetch_tables():
    conn = None
    try:
        # ✅ Connect to PostgreSQL using asyncpg
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        print("✅ Connected to database!")

        # ✅ Fetch list of tables
        rows = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)

        print("🗃 Tables in database:")
        for row in rows:
            print("-", row['table_name'])

    except Exception as e:
        print("❌ Error:", e)

    finally:
        if conn:
            await conn.close()
            print("🔌 Database connection closed.")

if __name__ == "__main__":
    asyncio.run(fetch_tables())
