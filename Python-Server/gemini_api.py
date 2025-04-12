import os
import logging
from google import genai
from dotenv import load_dotenv

# Configure logging for production use.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv_path = r"D:\Human Centered NLP\conversation-ai\server\.env"
load_dotenv(dotenv_path)

# Retrieve the API key from environment variables.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")

# Initialize the Google generative AI client.
client = genai.Client(api_key=GEMINI_API_KEY)


def query_gemini_api(prompt: str, model: str = "gemini-2.0-flash") -> str:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        if not response.text:
            raise ValueError("Gemini API returned no text.")
        logger.info("Gemini API call succeeded.")
        return response.text
    except Exception as e:
        logger.error(f"Error querying Gemini API: {e}")
        raise
