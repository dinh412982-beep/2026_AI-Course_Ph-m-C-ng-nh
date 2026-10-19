FROM python:3.11-slim

WORKDIR /app

# build-essential cần cho một số wheel của torch/sentence-transformers trên vài nền tảng
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài torch bản CPU-only riêng TRƯỚC, từ index chính thức của PyTorch.
# Bản mặc định trên PyPI kéo theo cả bộ thư viện CUDA cho GPU (nvidia_cublas,
# nvidia_cudnn, ...) nặng thêm 2-4 GB dù backend này chỉ chạy CPU, không
# dùng GPU. Cài riêng bản CPU-only giúp build nhanh hơn RẤT nhiều.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 1 worker vì model SBERT được load vào RAM một lần khi khởi động process.
# Muốn nhiều worker hơn thì cân nhắc dùng --preload hoặc tách model ra service riêng.
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "1", "--timeout", "120", "app:app"]
