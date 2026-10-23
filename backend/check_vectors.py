import os
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize

conn = psycopg2.connect(
    host=os.getenv("HOST", "db"),
    port=os.getenv("PORT", "5432"),
    dbname=os.getenv("POSTGRES_DB", "luat_db"),
    user=os.getenv("POSTGRES_USER", "chatbot"),
    password=os.getenv("POSTGRES_PASSWORD", "chatbot"),
)
cur = conn.cursor()
cur.execute("SELECT id, content, embedding::text FROM legal_documents WHERE id IN (1, 65, 104, 201);")
rows = cur.fetchall()
cur.close(); conn.close()

model = SentenceTransformer("keepitreal/vietnamese-sbert")
print(f"Số chiều model: {model.get_sentence_embedding_dimension()}")

for rid, content, emb_text in rows:
    stored = np.array([float(x) for x in emb_text.strip("[]").split(",")])
    fresh = model.encode(word_tokenize(content, format="text"))
    cos = float(np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh)))
    print(f"id={rid:<4} dim_db={len(stored)} cos(stored, encode_lai)={cos:.4f} | {content[:60]}")