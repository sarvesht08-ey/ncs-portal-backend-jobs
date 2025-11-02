import asyncio
import asyncpg

async def test_connection():
    DB_URL = "postgresql://postgres:root@localhost:5432/ncs_db"
    try:
        conn = await asyncpg.connect(DB_URL)
        print("Connection successful!")
        await conn.close()
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
