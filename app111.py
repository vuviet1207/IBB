# app.py
from flask import Flask, request, render_template, send_file, url_for
from bk import run_inference
import numpy as np
import cv2
import os
import time

app = Flask(__name__)

# Lưu đường dẫn ảnh kết quả gần nhất (đơn giản cho demo)
LATEST_RESULT_PATH = os.path.join("check", "test_result.jpg")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    global LATEST_RESULT_PATH

    file = request.files.get('image')
    if not file:
        return render_template("error.html",
                               message="❌ Không có file nào được tải lên.",
                               back_url=url_for('index')), 400

    # Đọc ảnh trực tiếp từ file upload (ndarray BGR)
    file_bytes = file.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        return render_template("error.html",
                               message="❌ Ảnh không hợp lệ hoặc bị lỗi.",
                               back_url=url_for('index')), 400

    # LẤY GIỚI TÍNH TỪ FORM
    gender = request.form.get('gender', 'male').lower()
    if gender not in ('male', 'female'):
        gender = 'male'

    # Gọi xử lý và TRUYỀN GIỚI TÍNH VÀO
    output_path, results = run_inference(img, test_image_path=None, gender=gender)

    if results.get("_error"):
        return render_template("error.html",
                               message=results["_error"],
                               back_url=url_for('index')), 422

    # Tổng hợp 'shoes' nếu chưa có
    if "shoes" not in results:
        left_shoe = results.get("left_shoe", "missing")
        right_shoe = results.get("right_shoe", "missing")
        results["shoes"] = "pass" if (left_shoe == "pass" and right_shoe == "pass") else "fail"

    # Chỉ hiển thị các khóa chính
    PUBLIC_KEYS = ("arms", "smile", "shoes", "legs", "scollar","eyebrow_hair","ear")
    results_public = {k: results.get(k) for k in PUBLIC_KEYS}

    if output_path:
        LATEST_RESULT_PATH = output_path

    img_url = url_for('result_image', v=int(time.time()))
    return render_template("result.html", img_path=img_url, results=results_public)

@app.route('/result')
def result_image():
    path = os.path.join("check", "test_result.jpg")
    if not os.path.exists(path):
        path = LATEST_RESULT_PATH
    if not os.path.exists(path):
        return "❌ Chưa có ảnh kết quả để hiển thị.", 404
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
