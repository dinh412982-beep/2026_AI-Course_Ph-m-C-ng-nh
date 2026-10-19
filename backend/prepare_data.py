"""
prepare_data.py
----------------
Bước 1 trong pipeline (bước còn thiếu trước import_data.py): đọc văn bản
luật gốc từ data/btl_nlp.docx, tách thành các dòng "tiêu đề nhóm" +
"Điều ..." giống cấu trúc đã kiểm chứng trong du_lieu_luat_sbert.pkl, rồi
TÁCH TỪ TIẾNG VIỆT bằng underthesea, encode bằng SBERT và lưu ra file
.pkl cùng cấu trúc:

    {
        "cac_doan_luat": [str, ...],   # text GỐC (chưa tách từ) để hiển thị
        "vector_luat":   ndarray[N, D],  # embedding từ text ĐÃ tách từ
    }

Đây là bước tái hiện đúng quy trình gốc đã dùng để tạo du_lieu_luat_sbert.pkl
(đã xác nhận qua notebook của người dùng): mỗi dòng được
underthesea.word_tokenize(..., format="text") trước khi đưa vào SBERT,
ví dụ "mũ bảo hiểm" -> "mũ_bảo_hiểm", giúp model PhoBERT-based hiểu đúng
từ ghép tiếng Việt. Bản text GỐC (chưa nối gạch dưới) vẫn được lưu lại
trong "cac_doan_luat" để hiển thị cho người dùng.

Chỉ cần chạy lại script này khi bạn CẬP NHẬT nội dung docx (thêm/sửa điều
luật). Nếu bạn đã có sẵn du_lieu_luat_sbert.pkl và không đổi nội dung, có
thể bỏ qua bước này và chạy thẳng import_data.py.

Định dạng docx mong đợi (mỗi paragraph là 1 dòng):
    - Dòng tiêu đề nhóm: không bắt đầu bằng "Điều <số>"
      ví dụ: "NHÓM 1: CÁC LỖI CỦA XE MÁY (MÔ TÔ, XE GẮN MÁY)"
    - Dòng nội dung: bắt đầu bằng "Điều <số> Khoản <số>: ..."

Nếu docx của bạn có cấu trúc khác (bảng biểu, nhiều dòng/1 điều, ...),
sửa lại hàm extract_lines_from_docx() bên dưới cho phù hợp.

Chạy:
    python prepare_data.py --docx ../data/btl_nlp.docx --out du_lieu_luat_sbert.pkl
"""

import argparse
import os
import pickle
import re

ARTICLE_PATTERN = re.compile(r"^Điều\s+\d+", re.IGNORECASE)


def extract_lines_from_docx(docx_path: str):
    """Đọc file docx, trả về list các dòng text (đã strip, bỏ dòng rỗng)."""
    from docx import Document  # python-docx

    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docx",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "btl_nlp.docx"),
        help="Đường dẫn tới file docx nguồn.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "du_lieu_luat_sbert.pkl"),
        help="Đường dẫn file .pkl output.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert"),
        help="Tên model SentenceTransformer dùng để encode.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.docx):
        raise FileNotFoundError(
            f"Không tìm thấy {args.docx}. Đặt btl_nlp.docx vào thư mục data/ "
            f"hoặc truyền --docx <đường dẫn>."
        )

    print(f"Đang đọc {args.docx} ...")
    lines = extract_lines_from_docx(args.docx)
    n_headers = sum(1 for l in lines if not ARTICLE_PATTERN.match(l))
    n_articles = len(lines) - n_headers
    print(f"-> {len(lines)} dòng ({n_headers} tiêu đề nhóm, {n_articles} điều luật).")

    print(f"Đang tải model SBERT '{args.model}' (cần internet lần đầu để tải model) ...")
    from sentence_transformers import SentenceTransformer
    from underthesea import word_tokenize

    model = SentenceTransformer(args.model)

    print("Đang tách từ tiếng Việt (underthesea) trước khi encode...")
    # QUAN TRỌNG: đây là bước đã xác nhận từ notebook gốc của người dùng -
    # SBERT được huấn luyện/encode trên text ĐÃ tách từ (vd "mũ bảo hiểm"
    # -> "mũ_bảo_hiểm"), không phải text thô. Bỏ bước này sẽ tạo ra vector
    # không tương thích với app.py.
    segmented_lines = [word_tokenize(line, format="text") for line in lines]

    print("Đang encode toàn bộ dòng đã tách từ (kể cả tiêu đề nhóm, giống cấu trúc gốc)...")
    vectors = model.encode(segmented_lines, show_progress_bar=True)

    # Lưu text GỐC (chưa tách từ) để hiển thị đẹp cho người dùng, giống
    # đúng cách notebook gốc làm ("Lưu bản gốc để hiển thị cho đẹp").
    data = {"cac_doan_luat": lines, "vector_luat": vectors}
    with open(args.out, "wb") as f:
        pickle.dump(data, f)

    print(f"Đã lưu {args.out} ({len(lines)} dòng, vector dim={vectors.shape[1]}).")
    print("Bước tiếp theo: chạy import_data.py để nạp vào Postgres.")


if __name__ == "__main__":
    main()
