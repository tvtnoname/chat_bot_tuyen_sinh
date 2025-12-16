import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.external.school_api import external_api_service

async def check_statuses():
    print("Fetching data...")
    data = await external_api_service.fetch_all_data()
    classes = data.get("classes", [])
    statuses = set()
    for c in classes:
        statuses.add(c.get("status"))
    
    print("Unique Statuses Found:", statuses)

if __name__ == "__main__":
    asyncio.run(check_statuses())
