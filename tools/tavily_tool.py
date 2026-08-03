from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()


def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


def tavily_search(query):
    client = get_tavily_client()
    if client is None:
        return (
            "Hotel search is unavailable because TAVILY_API_KEY is missing. "
            "Please add your Tavily API key to the .env file."
        )

    try:
        response = client.search(query=query, max_results=5)
    except Exception as exc:
        return f"Hotel search failed: {exc}"

    results = []

    for i, r in enumerate(response.get("results", []), 1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)
