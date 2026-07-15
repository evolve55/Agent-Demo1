import asyncio
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

async def test_connection():
    uri = os.getenv("SUPABASE_DB_URI")
    if not uri:
        print("❌ 错误：请先在 .env 文件里设置 SUPABASE_DB_URI")
        return

    try:
        conn = await asyncpg.connect(uri)
        version = await conn.fetchval("SELECT version();")
        print("✅ 连接成功！PostgreSQL 版本信息：")
        print(version)
        await conn.close()
    except Exception as e:
        print(f"❌ 连接失败：{e}")

if __name__ == "__main__":
    asyncio.run(test_connection())