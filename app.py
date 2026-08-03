from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import run_travel_agent
import pycountry

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None
    country: str = "India"
    transport: str = "All"
    start_date: str | None = None
    # `from_location` removed from the UI and backend — travel queries should be destination/start-date focused



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Build a sorted list of country names for the dropdown
    countries = [c.name for c in pycountry.countries]
    countries.sort()

    # Place India first in the list for convenience
    if "India" in countries:
        countries.insert(0, countries.pop(countries.index("India")))

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"countries": countries}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
            country=request_data.country,
            transport=request_data.transport,
            start_date=request_data.start_date,
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    import os
    import time

    # Prefer explicit PORT env var, otherwise default to 8000
    base_port = int(os.getenv("PORT", "8000"))

    # Try a few consecutive ports if the preferred one is taken
    max_tries = 5
    import socket

    for offset in range(max_tries):
        port = base_port + offset

        # Quick availability check: try to bind a temporary socket.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
        except OSError:
            print(f"Port {port} appears to be in use; trying next port.")
            if offset < max_tries - 1:
                time.sleep(0.2)
                continue
            else:
                print("Failed to find an available port in the attempted range.")
                raise

        # If we reach here, the port is available — start uvicorn on it.
        print(f"Starting server on port {port}...")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
        break