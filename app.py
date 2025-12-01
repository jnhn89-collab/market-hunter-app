from flask import Flask, render_template, request, jsonify, send_from_directory
from hunter import run_hunter
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 메인 화면
@app.route('/')
def home():
    return render_template('index.html')

# 2. 분석 요청 API
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        mode = data.get('mode', 'ROCKET')
        result = run_hunter(mode)
        return jsonify(result)
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🚨 [추가됨] 파일 제공 허용 (이게 없어서 404가 떴던 겁니다!)
@app.route('/database.json')
def get_db():
    return send_from_directory(BASE_DIR, 'database.json')

@app.route('/history.json')
def get_history():
    return send_from_directory(BASE_DIR, 'history.json')

@app.route('/daily_chart.png')
def get_chart():
    return send_from_directory(BASE_DIR, 'daily_chart.png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)