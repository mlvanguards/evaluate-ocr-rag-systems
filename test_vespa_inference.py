from src.ocr_benchmark.engines.vespa.retrieval.run import run_queries


async def main():
    """Main entry point for the application."""
    config_path = "/Users/vesaalexandru/Workspaces/cube/cube-publication/evaluate-ocr-rag-systems/src/ocr_benchmark/engines/vespa/vespa_config.yaml"

    queries = [
        "Percentage of non-fresh water as source?",
        "Policies related to nature risk?",
        "How much of produced water is recycled?",
    ]

    try:
        await run_queries(config_path, queries)
    except Exception:
        print("sss")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
