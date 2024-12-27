import asyncio

from vespa.deployment import VespaCloud
from vespa.io import VespaQueryResponse

from vespa_vision_rag import VespaSetup

cloud = VespaSetup("test")


async def query_vespa(queries, vespa_cloud):
    """
    Query Vespa for each user query.
    """
    app = vespa_cloud.get_application()

    async with app.asyncio(connections=1, timeout=180) as session:
        for query in queries:
            response: VespaQueryResponse = await session.query(
                yql="select title,url,image,page_number from pdf_page where userInput(@userQuery)",
                ranking="default",
                userQuery=query,
                timeout=120,
                hits=5,  # Adjust the number of hits returned
                body={"presentation.timing": True},
            )
            if response.is_successful():
                display_query_results(query, response)
            else:
                print(f"Query failed for '{query}': {response.json()}")


def display_query_results(query, response):
    """
    Display query results in a readable format.
    """
    query_time = response.json.get("timing", {}).get("searchtime", -1)
    total_count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    print(f"Query: {query}")
    print(f"Query Time: {query_time}s, Total Results: {total_count}")
    for idx, hit in enumerate(response.hits):
        title = hit["fields"].get("title", "N/A")
        url = hit["fields"].get("url", "N/A")
        page_number = hit["fields"].get("page_number", "N/A")
        print(f"  Result {idx + 1}:")
        print(f"    Title: {title}")
        print(f"    URL: {url}")
        print(f"    Page: {page_number}")


async def main():
    # Define the queries you want to execute
    queries = [
        "Percentage of non-fresh water as source?",
        "Policies related to nature risk?",
        "How much of produced water is recycled?",
    ]

    # Initialize VespaCloud
    vespa_cloud = VespaCloud(
        tenant="cube-digital",
        application="test",
        application_package=cloud.app_package,  # Use None since the app is already deployed
    )

    # Run queries against the Vespa app
    await query_vespa(queries, vespa_cloud)


if __name__ == "__main__":
    asyncio.run(main())
