
from dotenv import load_dotenv
import os
load_dotenv()
print("API Key loaded:", os.getenv("OPENAI_API_KEY")[:10] + "...")