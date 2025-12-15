import asyncio
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.external.school_api import external_api_service

async def main():
    logging.basicConfig(level=logging.INFO)
    print("--- TESTING REAL API INTEGRATION ---")
    
    # 1. Fetch Data
    print("\n1. Fetching all data...")
    data = await external_api_service.fetch_all_data()
    print(f"Fetch success: {'Yes' if data else 'No'}")
    if not data:
        print("Failed to fetch data. Exiting.")
        return

    print(f"Total Branches: {len(data.get('branches', []))}")
    print(f"Total Classes: {len(data.get('classes', []))}")

    # 2. Test Filtering (Based on sample data observations)
    # Branch: "Trung tâm Thăng Long"
    # Grade: "10" or "Lớp 10"
    b_query = "Thăng Long"
    g_query = "10"
    
    print(f"\n2. Filtering for Branch='{b_query}' and Grade='{g_query}'...")
    result = await external_api_service.get_filtered_data(b_query, g_query)
    
    if "message" in result and "classes_found" not in result:
        print("Filter Result Message:", result["message"])
    else:
        print("--- Filter Result ---")
        print("Context:", result.get("query_context"))
        print(f"Found {len(result.get('classes_found', []))} classes.")
        if result.get("classes_found"):
            print("Sample Class:", result["classes_found"][0])
        
        print(f"Found {len(result.get('teachers', []))} teachers.")
        if result.get("teachers"):
            print("Sample Teacher:", result["teachers"][0])
        
        print("Holidays:", result.get("holidays")[:3])

    # 3. Test Invalid Case
    print(f"\n3. Filtering for Invalid Branch...")
    inv_result = await external_api_service.get_filtered_data("InvalidBranch", "10")
    print("Result:", inv_result)

if __name__ == "__main__":
    asyncio.run(main())
