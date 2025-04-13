import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

dotenv_path = "../server/.env"
load_dotenv(dotenv_path)

api_key = os.getenv("SERPAPI_KEY")

def get_answer_box_and_top_organic_results(query):
    try:
        params = {
            "q": query,
            "api_key": api_key,
            "num": 20  # Requesting the top 20 organic results
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        answer_box = results.get("answer_box")
        organic_results = results.get("organic_results", [])

        return {
            "answer_box": answer_box,
            "organic_results": organic_results
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
