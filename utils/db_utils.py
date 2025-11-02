from typing import List, Dict
import asyncpg
import logging
import os
from config.settings import DB_URL as CONFIG_DB_URL

logger = logging.getLogger(__name__)
DB_URL = os.getenv("DATABASE_URL", CONFIG_DB_URL)  # Use env var if available, else fallback to config

async def get_complete_job_details(job_ids: List[str], location: str = None) -> List[Dict]:
    """Fetch complete job details with optional location filter"""
    if not job_ids:
        return []

    try:
        conn = await asyncpg.connect(DB_URL)
        try:
            if location:
                rows = await conn.fetch("""
                    SELECT *
                    FROM vacancies_summary
                    WHERE ncspjobid = ANY($1)
                      AND (LOWER(statename) = LOWER($2) OR LOWER(districtname) = LOWER($2))
                    ORDER BY ncspjobid;
                """, job_ids, location)
            else:
                rows = await conn.fetch("""
                    SELECT *
                    FROM vacancies_summary
                    WHERE ncspjobid = ANY($1)
                    ORDER BY ncspjobid;
                """, job_ids)

            return [dict(row) for row in rows]
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Failed to fetch job details: {e}")
        return []
