# English Learning Assistant

Một ứng dụng trợ lý học tiếng Anh thông minh sử dụng Google Gemini AI.

## Tính năng

- Tạo lộ trình học tập cá nhân hóa
- Kiểm tra và giải thích ngữ pháp
- Tạo bài đọc hiểu
- Học từ vựng theo chủ đề
- Luyện tập phát âm và giao tiếp

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/english-learning-assistant.git
cd english-learning-assistant
```

2. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Tạo file `.env` và thêm API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Sử dụng

1. Khởi động server:
```bash
uvicorn main:app --reload
```

2. Truy cập API tại: `http://localhost:8000`

## API Endpoints

- `POST /api/process`: Xử lý yêu cầu học tập
- `GET /api/health`: Kiểm tra trạng thái server

## Cấu trúc Project

```
english-learning-assistant/
├── agents/                 # Các agent AI
├── api/                    # API endpoints
├── config/                 # Cấu hình
├── tests/                  # Unit tests
├── .env                    # Environment variables
├── .gitignore             # Git ignore file
├── main.py                # Entry point
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## License

MIT License 