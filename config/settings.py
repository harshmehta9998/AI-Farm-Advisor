"""
config/settings.py
------------------
Central configuration for Agri-Edge.
Load API keys from environment variables (or .env via python-dotenv).
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_openweather_api_key_here")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

# ── Model Settings ────────────────────────────────────────────────────────────
MODEL_INPUT_SIZE    = (224, 224)          # MobileNetV2 input shape
MODEL_PATH          = "models/saved_model/plant_disease_model.h5"
CLASS_LABELS_PATH   = "models/saved_model/class_labels.json"

# ── App Settings ──────────────────────────────────────────────────────────────
APP_TITLE           = "Agri-Edge | AI Farm Advisor"
DEFAULT_CROP        = "Tomato"
SUPPORTED_CROPS     = ["Tomato", "Potato", "Rice", "Wheat", "Cotton", "Maize"]

# ── LLM Settings ─────────────────────────────────────────────────────────────
GROQ_MODEL          = "llama3-8b-8192"   # Fast, free-tier Groq model
LLM_MAX_TOKENS      = 700
LLM_TEMPERATURE     = 0.4
