import os
import vecs
import asyncio
from dotenv import load_dotenv

load_dotenv()

vx = None
docs = None

def initialize_db():
    global vx, docs
    vx = vecs.create_client(os.getenv("DB_CONNECTION"))
    docs = vx.get_or_create_collection(name="embeddings", dimension=1024)
    print(docs)

def check_db_ready():
    try:
        if vx is None:
            initialize_db()
        vx.list_collections()
        return True
    except Exception as e:
        print(f"DB not ready: {e}")
        return False

async def wait_for_db():
    while not check_db_ready():
        await asyncio.sleep(1)
    print("DB is ready")