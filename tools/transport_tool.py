import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_train_info(start: str | None = None, destination: str | None = None) -> str:
    host = os.getenv("RAIL_API_HOST")
    key = os.getenv("RAIL_API_KEY")

    if not host or not key:
        return "Train API is not configured. Please add RAIL_API_HOST and RAIL_API_KEY to your .env file."

    url = "https://indian-railways-train-fetcher.p.rapidapi.com/get_train_info"
    params = {
        "start": start or "None",
        "destination": destination or "None",
    }
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": host,
        "x-rapidapi-key": key,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        return str(data)
    except Exception as exc:
        return f"Train API request failed: {exc}"


def get_bus_info(location: str | None = None) -> str:
    host = os.getenv("BUS_API_HOST")
    key = os.getenv("BUS_API_KEY")

    if not host or not key:
        return "Bus API is not configured. Please add BUS_API_HOST and BUS_API_KEY to your .env file."

    url = "https://tripadvisor-com1.p.rapidapi.com/hotels/search"
    params = {"geoId": 60763}
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": host,
        "x-rapidapi-key": key,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        return str(data)
    except Exception as exc:
        return f"Bus API request failed: {exc}"
