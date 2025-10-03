import os
import vecs
from dotenv import load_dotenv

load_dotenv()

vx = vecs.create_client(os.getenv("DB_CONNECTION"))

docs = vx.get_or_create_collection(name="embeddings", dimension=1024)

print(docs)