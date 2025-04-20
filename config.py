import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
LLM_PROVIDER = "gemini"  # Options: "anthropic", "openai", or "gemini"

# Model settings
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
OPENAI_MODEL = "gpt-4o-2024-11-20"
GEMINI_MODEL = "gemini-2.0-flash-exp"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")