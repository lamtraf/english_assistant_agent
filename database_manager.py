import sqlite3
import json
import logging
from datetime import datetime

DATABASE_NAME = "english_learner_history.db"

logger = logging.getLogger(__name__)

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Cho phép truy cập cột bằng tên
    return conn

def initialize_database():
    """Khởi tạo CSDL và bảng nếu chưa tồn tại."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_input_type TEXT, -- 'text', 'audio_path', 'command_params'
            user_input_content TEXT, -- Nội dung text, đường dẫn file, hoặc JSON của params
            ai_response_type TEXT, -- 'text', 'audio_path', 'json_result'
            ai_response_content TEXT, -- Nội dung text, đường dẫn file, hoặc JSON của kết quả
            duration_ms INTEGER, -- Thời gian xử lý (tùy chọn)
            metadata TEXT -- JSON string cho các thông tin khác (ví dụ: streaming, model, v.v.)
        )
        """)
        conn.commit()
        logger.info(f"CSDL '{DATABASE_NAME}' đã được khởi tạo/kiểm tra thành công.")
    except sqlite3.Error as e:
        logger.error(f"Lỗi khi khởi tạo CSDL: {e}")
    finally:
        if conn:
            conn.close()

def log_interaction(
    user_id: str,
    agent_name: str,
    user_input_type: str,
    user_input_content: Any,
    ai_response_type: str,
    ai_response_content: Any,
    duration_ms: Optional[int] = None,
    metadata: Optional[dict] = None
):
    """Ghi một lượt tương tác vào CSDL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Chuyển đổi content và metadata thành JSON string nếu chúng là dict/list
        if isinstance(user_input_content, (dict, list)):
            user_input_content_str = json.dumps(user_input_content)
        else:
            user_input_content_str = str(user_input_content)

        if isinstance(ai_response_content, (dict, list)):
            ai_response_content_str = json.dumps(ai_response_content)
        else:
            ai_response_content_str = str(ai_response_content)
        
        metadata_str = json.dumps(metadata) if metadata else None

        cursor.execute("""
        INSERT INTO interactions (
            user_id, agent_name, user_input_type, user_input_content, 
            ai_response_type, ai_response_content, duration_ms, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, agent_name, user_input_type, user_input_content_str, 
              ai_response_type, ai_response_content_str, duration_ms, metadata_str))
        
        conn.commit()
        logger.info(f"Đã log tương tác cho user '{user_id}' với agent '{agent_name}'.")
    except sqlite3.Error as e:
        logger.error(f"Lỗi khi log tương tác vào CSDL: {e}")
    except Exception as ex:
        logger.error(f"Lỗi không mong muốn khi log tương tác: {ex}")
    finally:
        if conn:
            conn.close()

def get_user_history(user_id: str, limit: int = 20) -> list[dict]:
    """Lấy lịch sử tương tác của một người dùng."""
    history = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, user_id, agent_name, timestamp, user_input_type, user_input_content, 
               ai_response_type, ai_response_content, duration_ms, metadata
        FROM interactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        for row in rows:
            interaction = dict(row) # Chuyển sqlite3.Row thành dict
            # Parse lại JSON string nếu cần
            try:
                if interaction['user_input_content'] and (interaction['user_input_type'] == 'command_params' or interaction['user_input_type'].startswith('json')):
                    interaction['user_input_content'] = json.loads(interaction['user_input_content'])
            except json.JSONDecodeError:
                logger.warning(f"Không thể parse user_input_content JSON cho id {interaction['id']}")
            try:
                if interaction['ai_response_content'] and (interaction['ai_response_type'] == 'json_result' or interaction['ai_response_type'].startswith('json')):
                    interaction['ai_response_content'] = json.loads(interaction['ai_response_content'])
            except json.JSONDecodeError:
                logger.warning(f"Không thể parse ai_response_content JSON cho id {interaction['id']}")
            try:
                if interaction['metadata']:
                    interaction['metadata'] = json.loads(interaction['metadata'])
            except json.JSONDecodeError:
                 logger.warning(f"Không thể parse metadata JSON cho id {interaction['id']}")
            history.append(interaction)
        
        logger.info(f"Đã lấy {len(history)} bản ghi lịch sử cho user '{user_id}'.")
    except sqlite3.Error as e:
        logger.error(f"Lỗi khi lấy lịch sử người dùng từ CSDL: {e}")
    finally:
        if conn:
            conn.close()
    return history

# Gọi initialize_database khi module này được import lần đầu
# Hoặc tốt hơn là gọi nó một cách rõ ràng khi ứng dụng FastAPI khởi động.
# initialize_database() 

if __name__ == '__main__':
    # Test thử các hàm
    print("Khởi tạo CSDL...")
    initialize_database()
    print("CSDL sẵn sàng.")

    print("\nLog một vài tương tác mẫu...")
    log_interaction(
        user_id="test_user_001",
        agent_name="SpeakingPracticeAgent",
        user_input_type="audio_path",
        user_input_content="/path/to/user_audio.wav",
        ai_response_type="audio_path_and_text",
        ai_response_content={"text": "Hello there!", "audio_path": "/path/to/ai_audio.mp3"},
        duration_ms=1234,
        metadata={"model_used": "gemini-1.5-flash", "streaming": True}
    )
    log_interaction(
        user_id="test_user_001",
        agent_name="VocabularyAgent",
        user_input_type="command_params",
        user_input_content={"command": "explain_word", "word": "ubiquitous"},
        ai_response_type="json_result",
        ai_response_content={"word": "ubiquitous", "meaning": "present everywhere"},
        duration_ms=500
    )
    log_interaction(
        user_id="test_user_002",
        agent_name="StudyPlanAgent",
        user_input_type="text",
        user_input_content="Tạo lộ trình học IELTS 2 tháng",
        ai_response_type="text",
        ai_response_content="Đây là lộ trình của bạn...",
        metadata={"difficulty": "intermediate"}
    )
    print("Đã log xong.")

    print("\nLấy lịch sử cho test_user_001:")
    history1 = get_user_history("test_user_001")
    for item in history1:
        print(item)

    print("\nLấy lịch sử cho test_user_002 (giới hạn 1):")
    history2 = get_user_history("test_user_002", limit=1)
    for item in history2:
        print(item)

    print("\nLấy lịch sử cho user_unknown (không có):")
    history3 = get_user_history("user_unknown")
    print(history3) 