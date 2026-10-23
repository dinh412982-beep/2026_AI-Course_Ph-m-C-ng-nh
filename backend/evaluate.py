import app

# Mỗi phần tử: (câu hỏi, tiền tố điều luật đúng - có thể là chuỗi hoặc tuple các khoản hợp lệ)
TESTS = [
    ("không chấp hành hiệu lệnh của đèn tín hiệu giao thông", ("Điều 7 Khoản 7 Điểm c", "Điều 6 Khoản 9 Điểm b", "Điều 10 Khoản 1 Điểm b", "Điều 8 Khoản 7 Điểm c")),
    ("xe máy không chấp hành đèn tín hiệu", "Điều 7 Khoản 7 Điểm c"),
    ("vượt đèn đỏ", ("Điều 7 Khoản 7 Điểm c", "Điều 6 Khoản 9 Điểm b", "Điều 9 Khoản 2 Điểm đ")),
    ("đi qua ngã tư khi đèn đỏ", "Điều 7 Khoản 7 Điểm c"),
    ("không đội mũ bảo hiểm bị phạt bao nhiêu", ("Điều 7 Khoản 2 Điểm h", "Điều 12 Khoản 5 Điểm b")),
    ("nồng độ cồn xe máy", ("Điều 7 Khoản 6 Điểm a", "Điều 7 Khoản 8 Điểm b", "Điều 7 Khoản 9 Điểm d")),
]


def match_gold(content: str, gold) -> bool:
    if isinstance(gold, (list, tuple)):
        return any(content.startswith(g) for g in gold)
    return content.startswith(gold)


def run(buff_on):
    old = app.KEYWORD_BUFF
    if not buff_on:
        app.KEYWORD_BUFF = 0.0
    out = []
    try:
        for q, gold in TESTS:
            res = app.search_similar(q, top_k=5)
            rank = next((i + 1 for i, r in enumerate(res) if match_gold(r["content"], gold)), None)
            top1 = res[0]["similarity"] if res else 0.0
            out.append((q, gold, rank, top1))
    finally:
        app.KEYWORD_BUFF = old
    return out


def metrics(rows):
    n = len(rows)
    r1 = sum(1 for r in rows if r[2] == 1) / n
    r2 = sum(1 for r in rows if r[2] and r[2] <= 2) / n
    r5 = sum(1 for r in rows if r[2] and r[2] <= 5) / n
    mrr = sum((1 / r[2]) if r[2] else 0 for r in rows) / n
    return r1, r2, r5, mrr


print(f"n={len(TESTS)} | TOP_K={app.TOP_K} CANDIDATE_K={app.CANDIDATE_K} KEYWORD_BUFF={app.KEYWORD_BUFF}\n")
for buff_on in (True, False):
    r1, r2, r5, mrr = metrics(run(buff_on))
    print(f"{'Có buff' if buff_on else 'Không buff':<10} Recall@1={r1:.3f} Recall@2={r2:.3f} Recall@5={r5:.3f} MRR={mrr:.3f}")

print("\nChi tiết (có buff):")
for q, gold, rank, sim in run(True):
    print(f"rank={str(rank):<5} sim1={sim:.3f} | {q[:45]}")