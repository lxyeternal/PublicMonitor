import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://hackforums.net/search.php?action=results&sid=0554b56ec0c90104ae579f8b82fc73ea&sortby=&order=desc",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())