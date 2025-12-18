import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Lấy URL từ biến môi trường, fallback nếu không có
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://admin:admin123@localhost:5432/chatbot_db")

# Hack: SQLAlchemy async engine cần giao thức postgresql+asyncpg
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Khởi tạo Async Engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Tạo Session Factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Lớp cơ sở (Base Class) cho các Models
Base = declarative_base()

async def get_db():
    """Dependency injection cho FastAPI."""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Khởi tạo bảng cơ sở dữ liệu (chỉ dùng trong môi trường phát triển)."""
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Bỏ comment nếu muốn reset Database
        await conn.run_sync(Base.metadata.create_all)
