# ================================
# AI NEWS → 16 TWEETS → TELEGRAM BOT
# ================================

import os
import requests
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph

# ================================
# 🔑 LOAD ENV VARIABLES
# ================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Validate keys
if not all([OPENAI_API_KEY, SERPAPI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("❌ Missing one or more environment variables")

# ================================
# 🧠 STATE DEFINITION
# ================================

class NewsState(TypedDict):
    news: List[Dict]
    formatted_news: str
    tweets: str


# ================================
# 🤖 LLM
# ================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6
)

# ================================
# 📰 FETCH NEWS (SERPAPI)
# ================================

def fetch_news(state: NewsState) -> NewsState:
    search = SerpAPIWrapper(params={
        "engine": "google_news",
        "hl": "en",
        "gl": "in"
    })

    query = "Artificial Intelligence OR IT Tech OR Machine Learning OR Data Science latest news"
    results = search.results(query)

    news_results = results.get("news_results", [])

    articles = []
    for item in news_results[:6]:
        articles.append({
            "title": item.get("title"),
            "source": item.get("source"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })

    return {**state, "news": articles}


# ================================
# 🧾 FORMAT NEWS
# ================================

def format_news(state: NewsState) -> NewsState:
    formatted = "\n\n".join([
        f"{n['title']} ({n['source']}): {n['snippet']}"
        for n in state["news"]
    ])

    return {**state, "formatted_news": formatted}


# ================================
# 🧠 GENERATE 16 TWEETS
# ================================

def generate_tweets(state: NewsState) -> NewsState:
    prompt = f"""
You are a top AI + Tech Twitter creator.

Generate EXACTLY 16 tweets.

STRUCTURE:
1-4 → Latest news (based on input)
5-10 → Tech insights/AI tools info/comparisons
11-14 → Memes (funny + relatable) 3 in english 1 in hinglish
15-16 → Interactive questions

RULES:
- Each tweet < 280 characters
- No numbering inside tweets
- Separate tweets using "----"
- Keep them engaging, modern, crisp
- Avoid repetition

NEWS:
{state['formatted_news']}
"""

    response = llm.invoke(prompt)

    return {**state, "tweets": response.content}


# ================================
# ✂️ VALIDATE + CLEAN OUTPUT
# ================================

def enforce_length(tweet, max_len=280):
    return tweet if len(tweet) <= max_len else tweet[:max_len-3] + "..."


def clean_tweets(raw_text):
    tweets = raw_text.split("----")
    tweets = [enforce_length(t.strip()) for t in tweets if t.strip()]

    if len(tweets) > 16:
        tweets = tweets[:16]

    return tweets


# ================================
# 📤 TELEGRAM (WITH SPLITTING)
# ================================

TELEGRAM_LIMIT = 4000  # safe limit (<4096)


def split_message(text, limit=TELEGRAM_LIMIT):
    parts = []
    while len(text) > limit:
        split_index = text.rfind("\n", 0, limit)
        if split_index == -1:
            split_index = limit
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts


def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    messages = split_message(message)

    for idx, msg in enumerate(messages):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown"
        }

        response = requests.post(url, data=payload)

        if response.status_code != 200:
            print(f"❌ Telegram Error (part {idx+1}):", response.text)
        else:
            print(f"✅ Sent part {idx+1}/{len(messages)}")


# ================================
# 🔗 BUILD LANGGRAPH PIPELINE
# ================================

graph = StateGraph(NewsState)

graph.add_node("fetch_news", fetch_news)
graph.add_node("format_news", format_news)
graph.add_node("generate_tweets", generate_tweets)

graph.set_entry_point("fetch_news")
graph.add_edge("fetch_news", "format_news")
graph.add_edge("format_news", "generate_tweets")

app = graph.compile()


# ================================
# 🚀 RUN PIPELINE
# ================================

def run():
    result = app.invoke({
        "news": [],
        "formatted_news": "",
        "tweets": ""
    })

    tweets = clean_tweets(result["tweets"])
    final_output = "\n\n".join(tweets)

    print("\n===== GENERATED TWEETS =====\n")
    print(final_output)

    send_to_telegram(final_output)


# ================================
# ▶️ EXECUTE
# ================================

if __name__ == "__main__":
    run()