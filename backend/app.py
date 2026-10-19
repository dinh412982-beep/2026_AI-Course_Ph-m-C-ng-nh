"""
app.py
------
Buoc 3-4 trong pipeline: nhan cau hoi tu frontend -> TACH TU bang
underthesea (giong het buoc tao du_lieu_luat_sbert.pkl) -> encode bang
SBERT 'keepitreal/vietnamese-sbert' -> tim cac doan luat gan nghia nhat
trong Postgres/pgvector -> buff diem theo tu khoa (giong notebook goc) ->
ghep thanh cau tra loi.

QUAN TRONG: du lieu trong du_lieu_luat_sbert.pkl duoc encode tu text DA
TACH TU (word_tokenize format="text"), khong phai text tho. Neu bo buoc
tach tu nay, vector cau hoi va vector database se khong cung khong gian
ngu nghia -> tim kiem sai. Xem backend/prepare_data.py de biet chi tiet
buoc tao du lieu goc.

Endpoint chinh:
    POST /api/chat   { "question": "..." }
    ->   { "answer": "...", "matches": [ {category, content, similarity}, ... ] }
"""

import os

import psycopg2
from flask import Flask, jsonify, request
from flask_cors import CORS
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://chatbot:chatbot@localhost:5432/luat_db"
)
# QUAN TRONG: model nay phai CUNG voi model da dung de tao
# du_lieu_luat_sbert.pkl (da xac nhan: keepitreal/vietnamese-sbert).
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert")

# Cac tham so nay lay dung gia tri da duoc test trong notebook goc cua
# nguoi dung (tim_kiem_nang_cao). Co the chinh qua bien moi truong.
TOP_K = int(os.environ.get("TOP_K", 2))
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", 0.45))
KEYWORD_BUFF = float(os.environ.get("KEYWORD_BUFF", 0.28))
BUFF_SCORE_CAP = float(os.environ.get("BUFF_SCORE_CAP", 0.98))
# So ung vien lay tu DB truoc khi buff/loc theo tu khoa. Notebook goc chi
# lay dung TOP_K ung vien roi moi buff (candidate_k = TOP_K). Co the tang
# CANDIDATE_K de buff co co hoi "vot" them ket qua dung nhung bi pgvector
# xep hang thap hon TOP_K ban dau -- day la cai tien tuy chon, khong bat
# buoc phai bat.
CANDIDATE_K = int(os.environ.get("CANDIDATE_K", TOP_K))

STOP_WORDS = {"gì", "bị", "cho", "của", "với", "như", "tại", "vào", "để", "và", "hay"}

app = Flask(__name__)
CORS(app)

print(f"[startup] Đang tải embedding model '{EMBEDDING_MODEL_NAME}' ...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("[startup] Model đã sẵn sàng.")


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def segment(text: str) -> str:
    """Tách từ tiếng Việt, đúng bước đã dùng khi tạo du_lieu_luat_sbert.pkl."""
    return word_tokenize(text, format="text")


def keyword_overlap_ratio(question_raw: str, content: str):
    """Trả (số từ khóa khớp, tổng số từ khóa) giữa câu hỏi gốc và nội dung.

    Giữ nguyên logic từ notebook gốc: lọc từ có độ dài >= 2 ký tự và
    không thuộc STOP_WORDS, rồi đếm số từ khóa xuất hiện trong nội dung.
    """
    keywords = [w for w in question_raw.lower().split() if len(w) >= 2 and w not in STOP_WORDS]
    if not keywords:
        return 0, 0
    matched = sum(1 for kw in keywords if kw in content.lower())
    return matched, len(keywords)


def search_similar(question_raw: str, top_k: int = TOP_K, candidate_k: int = CANDIDATE_K):
    query_vec = model.encode(segment(question_raw))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT category, content, 1 - (embedding <=> %s) AS similarity
                FROM legal_documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, query_vec, candidate_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for category, content, score in rows:
        matched, total = keyword_overlap_ratio(question_raw, content)
        if total > 0:
            ratio = matched / total
            if ratio >= 0.5 or (total == 1 and matched == 1):
                score = min(score + KEYWORD_BUFF, BUFF_SCORE_CAP)
        results.append({"category": category, "content": content, "similarity": float(score)})

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


def build_answer(matches: list) -> str:
    filtered = [m for m in matches if m["similarity"] > MIN_SIMILARITY]
    if not filtered:
        return "Không tìm thấy điều luật nào phù hợp với câu hỏi của bạn."

    lines = [
        f"📌 [Độ chính xác: {m['similarity']:.0%}] {m['content']}"
        for m in filtered
    ]
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Trả lời sẵn cho các câu xã giao thông thường (chào hỏi, cảm ơn, hỏi bot là
# ai...) thay vì đưa vào tìm kiếm vector - tránh việc SBERT "đoán bừa" ra 1
# điều luật không liên quan chỉ vì cosine similarity tình cờ vượt ngưỡng.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Trả lời sẵn cho các câu xã giao thông thường (chào hỏi, cảm ơn, hỏi bot là
# ai, hỏi về chức năng website...) theo quy chuẩn Trợ lý AI Giao Thông
# ---------------------------------------------------------------------------
GREETINGS_EXACT = {
    "alo", "chao", "chào", "hi", "hello", "hey",
    "xin chao", "xin chào", "hi bot", "chao bot", "chào bot",
    "hello bot", "chao ban", "chào bạn",
}
GREETINGS_PREFIX = (
    "xin chào", "xin chao", "chào bạn", "chao ban",
    "chào bot", "chao bot", "hi bot", "hello bot", "alo bot",
)

FUNCTION_QUERIES = (
    "web nay lam duoc gi", "web này làm được gì",
    "website nay lam duoc gi", "website này làm được gì",
    "web co chuc nang gi", "web có chức năng gì",
    "website co chuc nang gi", "website có chức năng gì",
    "ban co chuc nang gi", "bạn có chức năng gì",
    "ban lam duoc gi", "bạn làm được gì",
    "chuc nang cua web", "chức năng của web",
    "chuc nang cua website", "chức năng của website",
    "chuc nang", "chức năng",
    "tinh nang", "tính năng",
    "huong dan su dung", "hướng dẫn sử dụng",
    "cach su dung", "cách sử dụng",
    "huong dan", "hướng dẫn",
)

THANKS = ("cam on", "cảm ơn", "thanks", "thank you")
BYE = ("tam biet", "tạm biệt", "bye", "goodbye")
WHO_ARE_YOU = ("ban la ai", "bạn là ai", "may la ai", "mày là ai", "ai day", "ai đấy")


def detect_smalltalk(question_raw: str):
    """Trả về câu trả lời sẵn nếu câu hỏi là xã giao thông thường, hoặc
    None nếu không khớp (lúc đó sẽ đi tiếp vào tìm kiếm vector như bình
    thường)."""
    q = question_raw.lower().strip().strip("!.,?~ ")

    # 1. Hỏi về chức năng website / bot
    if any(fq in q for fq in FUNCTION_QUERIES):
        return (
            "Hệ thống giúp bạn tra cứu nhanh luật giao thông mà không cần tài khoản. Bạn có thể sử dụng các chức năng sau:\n"
            "- Nhập trực tiếp tình huống vi phạm vào khung chat để tôi tra cứu mức phạt.\n"
            "- Xem mục **Mức phạt phổ biến** ở thanh công cụ để cập nhật nhanh các lỗi thường gặp (nồng độ cồn, tốc độ...).\n"
            "- Xem mục **Văn bản pháp luật** để đọc toàn văn Nghị định 100, Nghị định 123 và các luật liên quan."
        )

    # 2. Chào hỏi
    is_greeting_exact = q in GREETINGS_EXACT
    is_greeting_start = any(q.startswith(prefix) for prefix in GREETINGS_PREFIX) and not any(
        kw in q for kw in ("phạt", "phat", "bao nhiêu", "bao nhieu", "luật", "luat", "xe", "lỗi", "loi")
    )
    if is_greeting_exact or is_greeting_start:
        return (
            "Xin chào! Tôi là Trợ lý AI tra cứu luật giao thông. "
            "Bạn đang cần kiểm tra mức phạt cho hành vi vi phạm nào (Ví dụ: vượt đèn đỏ ô tô, nồng độ cồn xe máy...)?"
        )

    # 3. Hỏi bot là ai
    if any(w in q for w in WHO_ARE_YOU):
        return (
            "Tôi là Trợ lý AI Giao Thông, hệ thống tư vấn và tra cứu luật giao thông đường bộ Việt Nam "
            "công khai, miễn phí cho người dân. Nhiệm vụ của tôi là giải đáp tình huống giao thông "
            "và cung cấp mức phạt chính xác dựa trên các văn bản pháp luật hiện hành."
        )

    # 4. Cảm ơn & Tạm biệt
    if any(t in q for t in THANKS):
        return "Không có gì, rất vui được hỗ trợ bạn! Bạn có cần kiểm tra thêm mức phạt cho hành vi nào khác không?"
    if q in BYE or any(b in q for b in BYE):
        return "Tạm biệt bạn! Chúc bạn tham gia giao thông an toàn và vạn dặm bình an."

    return None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Thiếu trường 'question' trong request."}), 400

    smalltalk_reply = detect_smalltalk(question)
    if smalltalk_reply:
        return jsonify({"answer": smalltalk_reply, "matches": []})

    try:
        matches = search_similar(question)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Lỗi khi truy vấn database")
        return jsonify({"error": f"Lỗi truy vấn dữ liệu: {exc}"}), 500

    answer = build_answer(matches)
    return jsonify({"answer": answer, "matches": matches})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    rating = payload.get("rating")
    comment = (payload.get("comment") or "").strip()
    category = (payload.get("category") or "Chung").strip()
    question = (payload.get("question") or "").strip()

    app.logger.info(
        "[FEEDBACK] Rating=%s, Category=%s, Question=%s, Comment=%s",
        rating, category, question, comment,
    )
    return jsonify({
        "status": "ok",
        "message": "Cảm ơn bạn đã đóng góp ý kiến để hoàn thiện Trợ lý Pháp luật Giao thông!",
    })


if __name__ == "__main__":
    # Chạy dev server (không dùng khi deploy thật, Dockerfile đã dùng gunicorn)
    app.run(host="0.0.0.0", port=8000, debug=True)
