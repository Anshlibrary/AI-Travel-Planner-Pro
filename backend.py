import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        return None

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# LLM
# =========================

llm = None
if GROQ_API_KEY:
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )
    except Exception as exc:
        print(f"Warning: could not initialize Groq client: {exc}")

# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    country: str
    transport: str
    start_date: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Context Helpers
# =========================

def enrich_user_query(
    user_input: str,
    country: str | None = None,
    transport: str | None = None,
    start_date: str | None = None,
) -> str:
    clean_input = (user_input or "").strip()
    country_pref = (country or "India").strip() or "India"
    transport_pref = (transport or "All").strip() or "All"
    start_date_pref = (start_date or "").strip()

    if not clean_input:
        clean_input = "Plan a travel itinerary"

    if country_pref.lower() == "global":
        country_clause = "Country preference: global (any city/country combination is acceptable)"
    else:
        country_clause = f"Country preference: {country_pref}"

    if transport_pref.lower() == "all":
        transport_clause = "Transport preference: flights, trains, and buses"
    else:
        transport_clause = f"Transport preference: {transport_pref}"

    date_clause = f"Start date: {start_date_pref}" if start_date_pref else "Start date: not specified"

    global_guidance = ""
    if country_pref.lower() == "global":
        global_guidance = (
            "Global mode guidance: suggest flexible international routes, mention visa/entry requirements briefly, "
            "recommend practical city pairs, and keep the plan broadly useful for any city-to-city or country-to-country journey."
        )

    return f"{clean_input} | {country_clause} | {transport_clause} | {date_clause} | {global_guidance}".strip()


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(
        query,
        country=state.get("country", "India")
    )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }




# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Country Preference:
{state.get('country', 'Global')}

Transport Preference:
{state.get('transport', 'Flights')}

Start Date:
{state.get('start_date', 'Not specified')}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.

Short travel guidance rules:
- If country is Global, keep the answer flexible for any city-to-city or country-to-country trip.
- Mention visa/entry requirements briefly when the destination is international.
- Highlight the best transport mode for the route and maintain a practical travel flow.
- Keep suggestions short, clear, and easy to scan.
"""

    if llm is None:
        fallback_text = (
            "Itinerary generation is unavailable because GROQ_API_KEY is missing. "
            "Please add your API key to the .env file to enable LLM-generated plans."
        )
        return {
            "itinerary": fallback_text,
            "messages": [AIMessage(content=fallback_text)],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Country Preference:
{state.get('country', 'Global')}

Transport Preference:
{state.get('transport', 'Flights')}

Start Date:
{state.get('start_date', 'Not specified')}
Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- If the country is Global, present the plan as a flexible global trip guide with city-to-city options.
- Mention visa/entry requirements briefly when the destination is international.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    if llm is None:
        fallback_text = (
            "The AI travel plan is unavailable because GROQ_API_KEY is missing. "
            "Add your Groq API key to the .env file to enable full response generation."
        )
        return {
            "messages": [AIMessage(content=fallback_text)],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

checkpointer = None
travel_graph = None

if DATABASE_URL:
    try:
        _conn = psycopg.connect(
            DATABASE_URL,
            autocommit=True,
            row_factory=dict_row
        )
        checkpointer = PostgresSaver(_conn)
        checkpointer.setup()
        travel_graph = graph.compile(checkpointer=checkpointer)
    except Exception as exc:
        print(f"Warning: could not initialize PostgreSQL checkpointer: {exc}")
        travel_graph = graph.compile()
else:
    print("Warning: DATABASE_URL not set. Continuing without PostgreSQL persistence.")
    travel_graph = graph.compile()



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    country: str | None = None,
    transport: str | None = None,
    start_date: str | None = None,
):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    enhanced_query = enrich_user_query(user_input, country, transport, start_date)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=enhanced_query)
            ],
            "user_query": enhanced_query,
            "country": country or "India",
            "transport": transport or "All",
            "start_date": start_date or "",
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
