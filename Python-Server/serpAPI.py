from serpapi import GoogleSearch


def get_answer_box_and_top_organic_results(query, api_key):
    if not api_key:
        raise ValueError("No API key available for web search.")

    params = {
        "q": query,
        "api_key": api_key,
        "num": 20
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    answer_box = results.get("answer_box")
    organic_results = results.get("organic_results", [])

    return {
        "answer_box": answer_box,
        "organic_results": organic_results
    }
