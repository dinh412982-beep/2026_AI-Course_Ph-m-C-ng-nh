import os
import psycopg2
from sentence_transformers import SentenceTransformer, util
from underthesea import word_tokenize
import re

prefix = re.compile(r"^Điều \d+ Khoản \d+ (Điểm \S+ )?:\s*")
stripped = [prefix.sub("", c) for c in contents]
mat2 = model.encode([word_tokenize(s, format="text") for s in stripped],
                    convert_to_tensor=True, batch_size=32)
for q in queries:
    qv = model.encode(word_tokenize(q, format="text"), convert_to_tensor=True)
    sims = util.cos_sim(qv, mat2.to(qv.device))[0].cpu()
    order = sims.argsort(descending=True).tolist()
    rank = next(i + 1 for i, idx in enumerate(order) if ids[idx] == GOLD)
    print(f"no_prefix  rank_gold={rank:<4} | {q[:45]}")

conn = psycopg2.connect(
    host=os.getenv("HOST", "db"),
    port=os.getenv("PORT", "5432"),
    dbname=os.getenv("POSTGRES_DB", "luat_db"),
    user=os.getenv("POSTGRES_USER", "chatbot"),
    password=os.getenv("POSTGRES_PASSWORD", "chatbot"),
)
cur = conn.cursor()
cur.execute("SELECT id, content, embedding::text FROM legal_documents ORDER BY id;")
rows = cur.fetchall()
cur.close(); conn.close()

import torch
ids = [r[0] for r in rows]
contents = [r[1] for r in rows]
mat = torch.tensor([[float(x) for x in r[2].strip("[]").split(",")] for r in rows])

model = SentenceTransformer("keepitreal/vietnamese-sbert")

queries = [
    "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "xe máy không chấp hành đèn tín hiệu",
    "vượt đèn đỏ",
    "đi qua ngã tư khi đèn đỏ",
]
GOLD = 65

for q in queries:
    for label, text in [("raw", q), ("tokenized", word_tokenize(q, format="text"))]:
        qv = model.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(qv, mat.to(qv.device))[0].cpu()
        order = sims.argsort(descending=True).tolist()
        rank = next(i + 1 for i, idx in enumerate(order) if ids[idx] == GOLD)
        top1 = ids[order[0]]
        print(f"{label:<10} rank_gold={rank:<4} sim_gold={sims[ids.index(GOLD)]:.3f} "
              f"top1_id={top1} | {q[:45]}")