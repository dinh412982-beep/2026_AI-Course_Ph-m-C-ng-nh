"""
import_data.py
---------------
Bước 2 trong pipeline: nạp dữ liệu luật (đã encode sẵn bằng SBERT, lưu trong
du_lieu_luat_sbert.pkl) vào Postgres (pgvector) để backend truy vấn.

Cấu trúc file pkl (đã kiểm tra):
    {
        "cac_doan_luat": [str, str, ...],   # list các dòng text
        "vector_luat":   Tensor[N, 768],    # embedding SBERT tương ứng
    }

Lưu ý quan trọng (đã kiểm chứng trực tiếp trên file pkl thật): "cac_doan_luat"
là danh sách TRỘN LẪN giữa:
    - dòng tiêu đề nhóm luật, ví dụ:
          "🛵 NHÓM 1: CÁC LỖI CỦA XE MÁY (MÔ TÔ, XE GẮN MÁY)"
    - dòng nội dung điều luật thật sự, luôn bắt đầu bằng "Điều ...", ví dụ:
          "Điều 6 Khoản 2: Người điều khiển xe máy ... (Từ khóa: ...)"

Khi kiểm tra trực tiếp file mẫu (89 dòng text, 89 vector), MỌI dòng — kể cả
tiêu đề nhóm — đều có 1 vector embedding tương ứng CÙNG INDEX (do lúc tạo
dữ liệu, người tạo encode nguyên cả danh sách, không lọc tiêu đề). Vì vậy
script này ghép "cac_doan_luat[i]" với "vector_luat[i]" theo đúng chỉ số i:
    - nếu dòng là tiêu đề (không khớp "Điều ..."): cập nhật category hiện
      tại, KHÔNG insert vào DB (vector của tiêu đề bị bỏ qua, vì nội dung
      "NHÓM 1: ..." không phải là câu trả lời hữu ích cho câu hỏi luật).
    - nếu dòng là "Điều ...": insert (category, content, vector_luat[i]).

Nếu bộ dữ liệu của bạn khác (số text != số vector, tức tiêu đề KHÔNG được
encode), script tự động chuyển sang chế độ "tiêu thụ tuần tự" (mỗi dòng
"Điều..." lấy embedding kế tiếp) và in cảnh báo.

Chạy:
    python import_data.py
hoặc trong Docker:
    docker compose exec backend python import_data.py
"""

import os
import pickle
import re
import sys

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

PKL_PATH = os.environ.get(
    "SBERT_PKL_PATH",
    os.path.join(os.path.dirname(__file__), "du_lieu_luat_sbert.pkl"),
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://chatbot:chatbot@localhost:5432/luat_db"
)

# Dòng nội dung điều luật luôn bắt đầu bằng "Điều <số>"
ARTICLE_PATTERN = re.compile(r"^Điều\s+\d+", re.IGNORECASE)


def load_pickle(path: str):
    """Nạp du_lieu_luat_sbert.pkl.

    Import torch trước khi gọi pickle.load vì file được lưu bằng
    torch.save (bên trong có torch.FloatTensor / torch.FloatStorage),
    nên cần torch cài sẵn để unpickle đúng lớp.
    """
    import torch  # noqa: F401  (bắt buộc để pickle nhận diện torch.FloatStorage)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy copy du_lieu_luat_sbert.pkl vào "
            f"thư mục backend/ (hoặc set biến môi trường SBERT_PKL_PATH)."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    texts = data["cac_doan_luat"]
    vectors = data["vector_luat"]

    # vectors có thể là torch.Tensor -> chuyển sang numpy float32
    if hasattr(vectors, "detach"):
        vectors = vectors.detach().cpu().numpy()
    vectors = np.asarray(vectors, dtype=np.float32)

    return texts, vectors


def align_texts_and_vectors(texts, vectors):
    """Ghép mỗi dòng "Điều ..." với embedding tương ứng, giữ lại category
    (tiêu đề nhóm) gần nhất cho mỗi dòng nội dung.

    Trả về list các tuple (category, content, embedding_vector).
    """
    rows = []
    current_category = None

    if len(texts) == len(vectors):
        # Trường hợp đã kiểm chứng trên dữ liệu thật: mỗi dòng text (kể cả
        # tiêu đề nhóm) có 1 vector cùng index -> ghép trực tiếp theo index.
        for line, vec in zip(texts, vectors):
            line = (line or "").strip()
            if not line:
                continue
            if ARTICLE_PATTERN.match(line):
                rows.append((current_category, line, vec))
            else:
                current_category = line
        return rows

    # Fallback: số text và số vector không khớp -> giả định tiêu đề nhóm
    # KHÔNG được encode, mỗi dòng "Điều..." tiêu thụ 1 vector kế tiếp theo
    # thứ tự xuất hiện.
    print(
        f"[WARN] len(cac_doan_luat)={len(texts)} != len(vector_luat)={len(vectors)}. "
        f"Chuyển sang chế độ ghép tuần tự (fallback)."
    )
    vec_idx = 0
    for line in texts:
        line = (line or "").strip()
        if not line:
            continue
        if ARTICLE_PATTERN.match(line):
            if vec_idx >= len(vectors):
                print(f"[WARN] Hết vector nhưng vẫn còn dòng 'Điều...': {line[:60]}...")
                continue
            rows.append((current_category, line, vectors[vec_idx]))
            vec_idx += 1
        else:
            current_category = line

    if vec_idx != len(vectors):
        print(
            f"[WARN] Đã dùng {vec_idx} vector nhưng file pkl có {len(vectors)} vector."
        )

    return rows


def ensure_embedding_dim(conn, dim: int):
    """Kiểm tra cột embedding trong DB có đúng số chiều với dữ liệu sắp insert không."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'legal_documents'::regclass
              AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
    if row and row[0] and row[0] != dim:
        print(
            f"[WARN] Cột embedding trong DB có số chiều khác ({row[0]}) so với "
            f"dữ liệu import ({dim}). Sửa VECTOR({dim}) trong db/init.sql rồi "
            f"'docker compose down -v' để tạo lại DB."
        )


def main():
    print(f"Đang nạp dữ liệu từ {PKL_PATH} ...")
    texts, vectors = load_pickle(PKL_PATH)
    print(f"-> {len(texts)} dòng text, ma trận vector shape={vectors.shape}")

    rows = align_texts_and_vectors(texts, vectors)
    print(f"-> Chuẩn bị insert {len(rows)} bản ghi (category, content, embedding).")

    if not rows:
        print("Không có dữ liệu để import, dừng lại.")
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        ensure_embedding_dim(conn, rows[0][2].shape[0])
        with conn, conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE legal_documents RESTART IDENTITY;")
            for category, content, embedding in rows:
                cur.execute(
                    """
                    INSERT INTO legal_documents (category, content, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (category, content, embedding.tolist()),
                )
        print(f"Import thành công {len(rows)} bản ghi vào bảng legal_documents.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
