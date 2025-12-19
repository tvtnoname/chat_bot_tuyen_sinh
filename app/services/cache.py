import os
# Quản lý Cache (Redis)
import json
import logging
import inspect
from functools import wraps
import redis

# Cấu hình Redis từ biến môi trường
REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

class RedisCache:
    def __init__(self):
        try:
            if REDIS_URL:
                self.client = redis.from_url(REDIS_URL, decode_responses=True)
                logging.info(f"Đã kết nối Redis qua REDIS_URL")
            else:
                self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0, decode_responses=True)
                logging.info(f"Đã kết nối Redis tại {REDIS_HOST}:{REDIS_PORT}")
            
            self.client.ping()
            self.enabled = True
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
        except Exception as e:
            logging.error(f"Lỗi kết nối Redis: {e}")
            self.enabled = False

    def get(self, key: str):
        if not self.enabled: return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logging.error(f"Redis get error: {e}")
        return None

    def set(self, key: str, value, ttl: int = 3600):
        if not self.enabled: return
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logging.error(f"Redis set error: {e}")

redis_cache = RedisCache()

def cache_result(ttl: int = 3600):
    """Decorator để lưu cache kết quả trả về của hàm (Hỗ trợ Async)."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Tạo key định danh duy nhất dựa trên module, tên hàm và tham số đầu vào
            try:
                # Lấy tên tham số
                arg_names = inspect.signature(func).parameters.keys()
                # Ánh xạ giá trị vào tên tham số
                args_dict = dict(zip(arg_names, args))
                args_dict.update(kwargs)
                
                # Loại bỏ 'self' nếu là method của class
                if 'self' in args_dict:
                    del args_dict['self']
                
                # Tạo key string: filename:func_name:args
                key = f"{func.__module__}:{func.__name__}:{str(sorted(args_dict.items()))}"
            except Exception:
                key = f"{func.__module__}:{func.__name__}:{str(args)}:{str(kwargs)}"

            # 1. Kiểm tra Cache
            cached_val = redis_cache.get(key)
            if cached_val is not None:
                logging.info(f"[CACHE HIT] {func.__name__} (Lấy từ Redis)")
                return cached_val

            # 2. Thực thi hàm gốc
            result = await func(*args, **kwargs)

            # 3. Lưu kết quả vào Cache
            if result:
                redis_cache.set(key, result, ttl)
                logging.info(f"[CACHE MISS] {func.__name__} -> Đã lưu cache mới")
            
            return result
        return wrapper
    return decorator
