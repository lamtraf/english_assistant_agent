from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
import json
import logging
from typing import Any, Optional, List

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình database
DATABASE_URL = "sqlite+aiosqlite:///english_learner_history.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Models
class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_input_type = Column(String)
    user_input_content = Column(Text)
    ai_response_type = Column(String)
    ai_response_content = Column(Text)
    duration_ms = Column(Integer)
    meta_data = Column(Text)

async def init_db():
    """Khởi tạo database và tạo bảng nếu chưa tồn tại"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database đã được khởi tạo thành công")

async def get_session() -> AsyncSession:
    """Lấy session database"""
    async with async_session() as session:
        yield session

async def log_interaction(
    user_id: str,
    agent_name: str,
    user_input_type: str,
    user_input_content: Any,
    ai_response_type: str,
    ai_response_content: Any,
    duration_ms: Optional[int] = None,
    metadata: Optional[dict] = None
):
    """Ghi một lượt tương tác vào database"""
    try:
        # Chuyển đổi content và metadata thành JSON string
        if isinstance(user_input_content, (dict, list)):
            user_input_content_str = json.dumps(user_input_content)
        else:
            user_input_content_str = str(user_input_content)

        if isinstance(ai_response_content, (dict, list)):
            ai_response_content_str = json.dumps(ai_response_content)
        else:
            ai_response_content_str = str(ai_response_content)
        
        metadata_str = json.dumps(metadata) if metadata else None

        async with async_session() as session:
            interaction = Interaction(
                user_id=user_id,
                agent_name=agent_name,
                user_input_type=user_input_type,
                user_input_content=user_input_content_str,
                ai_response_type=ai_response_type,
                ai_response_content=ai_response_content_str,
                duration_ms=duration_ms,
                meta_data=metadata_str
            )
            session.add(interaction)
            await session.commit()
            
        logger.info(f"Đã log tương tác cho user '{user_id}' với agent '{agent_name}'")
    except Exception as e:
        logger.error(f"Lỗi khi log tương tác: {e}")
        raise

async def get_user_history(user_id: str, limit: int = 20) -> List[dict]:
    """Lấy lịch sử tương tác của một người dùng"""
    try:
        async with async_session() as session:
            query = session.query(Interaction).filter(
                Interaction.user_id == user_id
            ).order_by(
                Interaction.timestamp.desc()
            ).limit(limit)
            
            result = await session.execute(query)
            interactions = result.scalars().all()
            
            history = []
            for interaction in interactions:
                item = {
                    "id": interaction.id,
                    "user_id": interaction.user_id,
                    "agent_name": interaction.agent_name,
                    "timestamp": interaction.timestamp.isoformat(),
                    "user_input_type": interaction.user_input_type,
                    "user_input_content": interaction.user_input_content,
                    "ai_response_type": interaction.ai_response_type,
                    "ai_response_content": interaction.ai_response_content,
                    "duration_ms": interaction.duration_ms,
                    "metadata": interaction.meta_data
                }
                
                # Parse JSON strings back to objects
                try:
                    if item["user_input_content"] and (item["user_input_type"] == "command_params" or item["user_input_type"].startswith("json")):
                        item["user_input_content"] = json.loads(item["user_input_content"])
                except json.JSONDecodeError:
                    logger.warning(f"Không thể parse user_input_content JSON cho id {item['id']}")
                    
                try:
                    if item["ai_response_content"] and (item["ai_response_type"] == "json_result" or item["ai_response_type"].startswith("json")):
                        item["ai_response_content"] = json.loads(item["ai_response_content"])
                except json.JSONDecodeError:
                    logger.warning(f"Không thể parse ai_response_content JSON cho id {item['id']}")
                    
                try:
                    if item["metadata"]:
                        item["metadata"] = json.loads(item["metadata"])
                except json.JSONDecodeError:
                    logger.warning(f"Không thể parse metadata JSON cho id {item['id']}")
                    
                history.append(item)
            
            logger.info(f"Đã lấy {len(history)} bản ghi lịch sử cho user '{user_id}'")
            return history
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử người dùng: {e}")
        raise

# Khởi tạo database khi module được import
import asyncio
asyncio.create_task(init_db())

if __name__ == '__main__':
    # Test thử các hàm
    print("Khởi tạo CSDL...")
    asyncio.run(init_db())
    print("CSDL sẵn sàng.")

    print("\nLog một vài tương tác mẫu...")
    asyncio.run(log_interaction(
        user_id="test_user_001",
        agent_name="SpeakingPracticeAgent",
        user_input_type="audio_path",
        user_input_content="/path/to/user_audio.wav",
        ai_response_type="audio_path_and_text",
        ai_response_content={"text": "Hello there!", "audio_path": "/path/to/ai_audio.mp3"},
        duration_ms=1234,
        metadata={"model_used": "gemini-1.5-flash", "streaming": True}
    ))
    asyncio.run(log_interaction(
        user_id="test_user_001",
        agent_name="VocabularyAgent",
        user_input_type="command_params",
        user_input_content={"command": "explain_word", "word": "ubiquitous"},
        ai_response_type="json_result",
        ai_response_content={"word": "ubiquitous", "meaning": "present everywhere"},
        duration_ms=500
    ))
    asyncio.run(log_interaction(
        user_id="test_user_002",
        agent_name="StudyPlanAgent",
        user_input_type="text",
        user_input_content="Tạo lộ trình học IELTS 2 tháng",
        ai_response_type="text",
        ai_response_content="Đây là lộ trình của bạn...",
        metadata={"difficulty": "intermediate"}
    ))
    print("Đã log xong.")

    print("\nLấy lịch sử cho test_user_001:")
    history1 = asyncio.run(get_user_history("test_user_001"))
    for item in history1:
        print(item)

    print("\nLấy lịch sử cho test_user_002 (giới hạn 1):")
    history2 = asyncio.run(get_user_history("test_user_002", limit=1))
    for item in history2:
        print(item)

    print("\nLấy lịch sử cho user_unknown (không có):")
    history3 = asyncio.run(get_user_history("user_unknown"))
    print(history3) 