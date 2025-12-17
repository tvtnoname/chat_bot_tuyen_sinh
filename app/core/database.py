import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Lấy URL từ biến môi trường, fallback nếu không có
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://admin:admin123@localhost:5432/chatbot_db")

# Hack: SQLAlchemy async engine cần schema là postgresql+asyncpg
# Và Supabase cần ssl=require
if DATABASE_URL:
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Nếu không phải localhost (tức là cloud DB), thêm user pass ssl nếu chưa có
    if "localhost" not in DATABASE_URL and "ssl=" not in DATABASE_URL:
        # Nếu đã có query param rồi (?) thì thêm &ssl=require, chưa có thì ?ssl=require
        if "?" in DATABASE_URL:
            DATABASE_URL += "&ssl=require"
        else:
            DATABASE_URL += "?ssl=require"

# Khởi tạo Async Engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Tạo Session Factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Base class cho Models
Base = declarative_base()

async def get_db():
    """Dependency injection cho FastAPI."""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Khởi tạo bảng (dùng trong dev only)."""
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Uncomment nếu muốn reset DB
        await conn.run_sync(Base.metadata.create_all)
