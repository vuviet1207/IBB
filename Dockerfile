FROM python:3.12-slim

# Cài vài package hệ thống cần cho opencv, mediapipe (tùy app bạn)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    git \
    libglib2.0-0 \       
    libsm6 \            
    libpq-dev \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Cài thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy đúng những phần cần của app
COPY models/ ./models/
COPY static/ ./static/
COPY templates/ ./templates/
COPY app.py bk.py ./

# 3) Chạy app (đổi port nếu trong app.py bạn dùng port khác)
# Nếu bạn dùng Flask dev server:
CMD ["python", "app.py"]
# Nếu dùng gunicorn thì chỉnh:
# CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
