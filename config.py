# SAVE AS .env LATER
HOST = "127.0.0.1"
PORT = "19530"
COL = "company_policies"


REDIS_URL = "redis://localhost:6379/0"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7   # optional TTL for session keys
MAX_HISTORY_MESSAGES = 20                # how many recent messages to load into memory


gemini_chat_model = "gemini-2.5"   # pick the chat model you want

API_KEY = "GEMINI_API_KEY"  # environment variable name for Gemini/Google API key