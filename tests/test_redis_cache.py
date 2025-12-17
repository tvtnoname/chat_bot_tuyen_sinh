import asyncio
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.cache import redis_cache, cache_result, RedisCache
from app.services.external.school_api import external_api_service

# Dummy async function to test decorator
@cache_result(ttl=10)
async def expensive_operation(x: int):
    print(f"  [EXEC] Running expensive calc for {x}...")
    await asyncio.sleep(1) # Simulate delay
    return x * x

async def test_redis():
    print(">>> 1. Checking Redis Connection...")
    if not redis_cache.enabled:
        print("ERROR: Redis is NOT enabled/connected.")
        return

    print(">>> 2. Test Get/Set...")
    redis_cache.set("test_key", {"foo": "bar"}, ttl=10)
    val = redis_cache.get("test_key")
    print(f"    Got from redis: {val}")
    assert val == {"foo": "bar"}
    
    print(">>> 3. Test Decorator (First Call - Should Sleep)...")
    start = time.time()
    res1 = await expensive_operation(5)
    dur1 = time.time() - start
    print(f"    Res1: {res1}, Time: {dur1:.2f}s")
    assert res1 == 25
    assert dur1 >= 1.0 # Should be slow

    print(">>> 4. Test Decorator (Second Call - Should be Immediate)...")
    start = time.time()
    res2 = await expensive_operation(5)
    dur2 = time.time() - start
    print(f"    Res2: {res2}, Time: {dur2:.2f}s")
    assert res2 == 25
    assert dur2 < 0.1 # Should be instant
    
    print(">>> 5. Test School API Caching...")
    # Call twice
    print("    Fetching branches (1st)...")
    b1 = await external_api_service.get_all_branches()
    print("    Fetching branches (2nd)...")
    b2 = await external_api_service.get_all_branches()
    assert b1 == b2

    print(">>> SUCCESS: Redis Caching is working perfectly!")

if __name__ == "__main__":
    asyncio.run(test_redis())
