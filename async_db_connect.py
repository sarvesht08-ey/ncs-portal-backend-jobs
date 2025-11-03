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

        data = await conn.fetch("""
            SELECT 
                ncspjobid, title, keywords, description,
                CASE WHEN date IS NOT NULL THEN TO_CHAR(date, 'YYYY-MM-DD') ELSE NULL END as date,
                organizationid, organization_name, numberofopenings,
                industryname, sectorname, functionalareaname,
                functionalrolename, aveexp, avewage, gendercode,
                highestqualification, statename, districtname
            FROM vacancies_summary
            WHERE ncspjobid = ANY($1::text[])
            ORDER BY ncspjobid;
        """, ['20V63-1550061073345J'])


        print(data)

    except Exception as e:
        print("❌ Error:", e)

    finally:
        if conn:
            await conn.close()
            print("🔌 Database connection closed.")

if __name__ == "__main__":
    asyncio.run(fetch_tables())
