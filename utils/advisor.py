"""
utils/advisor.py
================
AI-Powered Agricultural Advisory Generator
-------------------------------------------
Generates structured, section-by-section farming advice for Indian smallholder
farmers by combining:
  - Crop disease detection results (from disease_detector.py)
  - Hyperlocal weather data (from weather_fetcher.py)
  - Mandi/market price data (from market_prices.py)
  - A rule-based risk scoring engine (no API key needed)
  - An LLM call (Groq or OpenAI) for natural language advisory text

PUBLIC FUNCTIONS
----------------
    generate_advisory(inputs, api_key, provider)  → AdvisoryResult dict
    compute_risk_score(inputs)                     → risk level + score + reasons
    format_advisory_markdown(result)               → human-readable markdown string

ADVISORY OUTPUT SCHEMA
-----------------------
{
  "risk_level"    : "HIGH",           # NONE | LOW | MODERATE | HIGH | CRITICAL
  "risk_score"    : 72,               # 0–100 composite score
  "risk_reasons"  : ["Late Blight detected at 91% confidence", ...],

  "sections": {
    "disease_warning"    : { "heading": str, "body": str, "urgency": str },
    "irrigation_advice"  : { "heading": str, "body": str },
    "crop_inputs"        : { "heading": str, "body": str },
    "market_advice"      : { "heading": str, "body": str },
    "action_plan"        : { "heading": str, "steps": [str, ...] },
  },

  "summary"       : str,              # One-sentence farmer-friendly summary
  "source"        : "groq|openai|mock",
  "model_used"    : "llama3-8b-8192",
  "generated_at"  : "2024-06-14 10:30:00 IST",
  "error"         : None,
}

SUPPORTED LLM PROVIDERS
-----------------------
  "groq"   — Groq API (recommended: free tier, ~800 tok/s)
             Sign up: https://console.groq.com
             Model  : llama3-8b-8192

  "openai" — OpenAI API
             Sign up: https://platform.openai.com
             Model  : gpt-4o-mini

Both use the same OpenAI-compatible /v1/chat/completions endpoint format.

API KEY SETUP
-------------
Add to your .env file:
    GROQ_API_KEY=your_groq_key_here
    OPENAI_API_KEY=your_openai_key_here   # optional alternative

CLI USAGE
---------
    python utils/advisor.py --crop Tomato --disease "Tomato - Early Blight" \\
                            --confidence 0.91 --pin 411001

    python utils/advisor.py --crop Potato --disease "Potato - Late Blight" \\
                            --confidence 0.95 --city Nashik --json

    python utils/advisor.py --crop Rice --no-disease --city Hyderabad \\
                            --provider openai

    python utils/advisor.py --risk-only --crop Tomato \\
                            --disease "Tomato - Early Blight" --confidence 0.87
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Provider endpoints (both are OpenAI-compatible)
PROVIDER_CONFIGS = {
    "groq": {
        "url"        : "https://api.groq.com/openai/v1/chat/completions",
        "model"      : "llama3-8b-8192",
        "max_tokens" : 900,
        "temperature": 0.35,
        "env_key"    : "GROQ_API_KEY",
        "placeholder": "your_groq_api_key_here",
    },
    "openai": {
        "url"        : "https://api.openai.com/v1/chat/completions",
        "model"      : "gpt-4o-mini",
        "max_tokens" : 900,
        "temperature": 0.35,
        "env_key"    : "OPENAI_API_KEY",
        "placeholder": "your_openai_api_key_here",
    },
}

DEFAULT_PROVIDER = "groq"

IST = timezone(timedelta(hours=5, minutes=30))

# ── Risk scoring weights (all sum to 100) ────────────────────────────────────
RISK_WEIGHTS = {
    "disease_confidence" : 35,   # How certain is the disease detection?
    "disease_severity"   : 25,   # How dangerous is this specific disease?
    "weather_stress"     : 20,   # Heat, humidity, frost conditions
    "market_loss_risk"   : 10,   # Is market price below cost of production?
    "rainfall_deficit"   : 10,   # Risk of drought / irrigation stress
}

# Disease severity scores (0–100). Maps disease label keywords → severity score.
# Higher = more dangerous / faster-spreading.
DISEASE_SEVERITY_MAP = {
    "late blight"      : 90,   # Phytophthora — extremely fast spread
    "early blight"     : 55,   # Alternaria — manageable with timely treatment
    "bacterial spot"   : 50,
    "leaf blast"       : 80,   # Rice blast — very destructive
    "brown spot"       : 45,
    "yellow rust"      : 75,   # Wheat rust — epidemic potential
    "brown rust"       : 60,
    "leaf mold"        : 35,   # Usually manageable
    "septoria"         : 50,
    "healthy"          : 0,    # No disease
}

# Risk level thresholds (composite 0–100 score → label)
RISK_THRESHOLDS = [
    (85, "CRITICAL"),
    (65, "HIGH"),
    (40, "MODERATE"),
    (20, "LOW"),
    (0,  "NONE"),
]

# Emoji badges for each section heading (displayed in Streamlit)
SECTION_EMOJIS = {
    "disease_warning"  : "🦠",
    "irrigation_advice": "💧",
    "crop_inputs"      : "🌱",
    "market_advice"    : "📊",
    "action_plan"      : "📋",
}


# ── Language configuration ────────────────────────────────────────────────────
# Maps UI label → internal config used in prompts and UI.
# Adding a new language: add an entry here — no other code changes needed.
LANGUAGE_CONFIG = {
    "English": {
        "code"         : "en",
        "native_name"  : "English",
        "flag"         : "🇬🇧",
        "instruction"  : "Write all advisory text in clear, simple English.",
        "json_note"    : "All JSON string values must be in English.",
        "ui": {
            "risk_label"     : "RISK",
            "score_label"    : "Score",
            "summary_prefix" : "📌 Summary",
            "plan_heading"   : "7-Day Action Plan",
            "demo_badge"     : "Demo Mode",
            "urgency_prefix" : "⏱ Urgency",
            "breakdown_label": "Risk Breakdown",
            "report_title"   : "Advisory Report",
            "risk_title"     : "Risk Assessment",
        },
    },
    "Hindi": {
        "code"         : "hi",
        "native_name"  : "हिन्दी",
        "flag"         : "🇮🇳",
        "instruction"  : (
            "Write ALL advisory text in simple, everyday Hindi (Devanagari script). "
            "Use language a village farmer in India would naturally understand. "
            "Keep sentences short. Use Hindi agricultural terms where natural "
            "(e.g. फसल, मंडी, सिंचाई, खाद, कीटनाशक). "
            "Do NOT mix English sentences into the Hindi text. "
            "Technical product names (Mancozeb, DAP, Urea) may stay in English."
        ),
        "json_note"    : (
            "IMPORTANT: All JSON string values (summary, headings, body text, steps, urgency) "
            "MUST be written in Hindi (Devanagari script). "
            "Only product/chemical names may remain in English."
        ),
        "ui": {
            "risk_label"     : "जोखिम",
            "score_label"    : "स्कोर",
            "summary_prefix" : "📌 सारांश",
            "plan_heading"   : "7-दिवसीय कार्य योजना",
            "demo_badge"     : "डेमो मोड",
            "urgency_prefix" : "⏱ तत्परता",
            "breakdown_label": "जोखिम विश्लेषण",
            "report_title"   : "सलाह रिपोर्ट",
            "risk_title"     : "जोखिम मूल्यांकन",
        },
    },
    "Kannada": {
        "code"         : "kn",
        "native_name"  : "ಕನ್ನಡ",
        "flag"         : "🇮🇳",
        "instruction"  : (
            "Write ALL advisory text in simple, everyday Kannada (Kannada script). "
            "Use language a farmer in Karnataka would naturally understand. "
            "Keep sentences short. Use Kannada agricultural terms where natural "
            "(e.g. ಬೆಳೆ, ಮಾರುಕಟ್ಟೆ, ನೀರಾವರಿ, ಗೊಬ್ಬರ, ಕೀಟನಾಶಕ). "
            "Do NOT mix English sentences. "
            "Technical product names (Mancozeb, DAP, Urea) may stay in English."
        ),
        "json_note"    : (
            "IMPORTANT: All JSON string values (summary, headings, body text, steps, urgency) "
            "MUST be written in Kannada (Kannada script). "
            "Only product/chemical names may remain in English."
        ),
        "ui": {
            "risk_label"     : "ಅಪಾಯ",
            "score_label"    : "ಅಂಕ",
            "summary_prefix" : "📌 ಸಾರಾಂಶ",
            "plan_heading"   : "7-ದಿನದ ಕಾರ್ಯ ಯೋಜನೆ",
            "demo_badge"     : "ಡೆಮೊ ಮೋಡ್",
            "urgency_prefix" : "⏱ ತುರ್ತು",
            "breakdown_label": "ಅಪಾಯ ವಿಶ್ಲೇಷಣೆ",
            "report_title"   : "ಸಲಹೆ ವರದಿ",
            "risk_title"     : "ಅಪಾಯ ಮೌಲ್ಯಮಾಪನ",
        },
    },
}

SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIG.keys())
DEFAULT_LANGUAGE     = "English"


def get_language_config(language: str) -> dict:
    """Return the config dict for a language, falling back to English."""
    return LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG[DEFAULT_LANGUAGE])


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _ist_now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _make_session() -> requests.Session:
    """Requests session with auto-retry on transient errors."""
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _get_disease_severity(disease_label: str) -> int:
    """
    Look up severity score for a disease label.
    Matches on lowercase substrings so 'Tomato - Late Blight' → 90.
    Returns 0 for healthy plants, 40 default for unknown diseases.
    """
    label_lower = (disease_label or "").lower()

    if "healthy" in label_lower:
        return 0

    for keyword, score in DISEASE_SEVERITY_MAP.items():
        if keyword in label_lower:
            return score

    return 40   # Unknown disease — assume moderate risk


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_to_level(score: int) -> str:
    for threshold, label in RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "NONE"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — RISK SCORING ENGINE (rule-based, no API needed)
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_score(inputs: dict) -> dict:
    """
    Compute a composite risk score (0–100) from all available farm inputs.
    This is rule-based — it runs instantly without any API call.

    Args:
        inputs: dict with keys:
            crop            (str)   — crop name
            disease_label   (str)   — e.g. "Tomato - Late Blight"
            confidence      (float) — model confidence 0.0–1.0
            weather         (dict)  — output of get_weather()
            market          (dict)  — output of get_market_prices()

    Returns:
        dict with:
            risk_score      (int)   — 0–100
            risk_level      (str)   — NONE | LOW | MODERATE | HIGH | CRITICAL
            component_scores (dict) — breakdown per factor
            risk_reasons    (list)  — human-readable explanations
            is_healthy      (bool)  — True if no disease detected
    """
    disease_label = inputs.get("disease_label", "") or ""
    confidence    = float(inputs.get("confidence", 0.0))
    weather       = inputs.get("weather") or {}
    market        = inputs.get("market")  or {}
    cur           = weather.get("current", weather)  # support both new and flat schema

    is_healthy    = "healthy" in disease_label.lower() or not disease_label
    reasons       = []
    components    = {}

    # ── Factor 1: Disease confidence (0–35 pts) ───────────────────────────────
    if is_healthy:
        disease_conf_score = 0
    else:
        # Scale: confidence × weight. High confidence + disease = high risk.
        disease_conf_score = int(confidence * RISK_WEIGHTS["disease_confidence"])
        if confidence >= 0.85:
            reasons.append(
                f"{disease_label} detected with HIGH confidence ({confidence*100:.0f}%) — act immediately."
            )
        elif confidence >= 0.60:
            reasons.append(
                f"{disease_label} detected ({confidence*100:.0f}% confidence) — inspect plants today."
            )
        else:
            reasons.append(
                f"Possible {disease_label} detected ({confidence*100:.0f}% confidence) — low certainty, monitor closely."
            )
    components["disease_confidence"] = disease_conf_score

    # ── Factor 2: Disease severity (0–25 pts) ────────────────────────────────
    severity_raw   = _get_disease_severity(disease_label)
    severity_score = int((severity_raw / 100) * RISK_WEIGHTS["disease_severity"])
    components["disease_severity"] = severity_score
    if severity_raw >= 75 and not is_healthy:
        reasons.append(f"{disease_label} is a fast-spreading disease — delay can cause 50–80% yield loss.")
    elif severity_raw >= 50 and not is_healthy:
        reasons.append(f"{disease_label} is moderately serious — treatment needed within 3–5 days.")

    # ── Factor 3: Weather stress (0–20 pts) ──────────────────────────────────
    temp     = float(cur.get("temp_c",       cur.get("temp",      28)))
    humidity = float(cur.get("humidity_pct", cur.get("humidity",  60)))
    wind_kmh = float(cur.get("wind_speed_kmh", 0))

    weather_stress = 0

    if temp >= 38:
        weather_stress += 10
        reasons.append(f"Extreme heat ({temp}°C) — severe crop stress, emergency irrigation may be needed.")
    elif temp >= 35:
        weather_stress += 6
        reasons.append(f"High temperature ({temp}°C) — heat stress risk for {inputs.get('crop', 'crop')}.")

    if temp <= 4:
        weather_stress += 10
        reasons.append(f"Near-frost temperature ({temp}°C) — cover sensitive crops tonight.")

    if humidity >= 85 and not is_healthy:
        weather_stress += 8
        reasons.append(f"Very high humidity ({humidity}%) will accelerate disease spread — spray immediately.")
    elif humidity >= 80 and not is_healthy:
        weather_stress += 5
        reasons.append(f"High humidity ({humidity}%) favours fungal/bacterial spread.")

    components["weather_stress"] = _clamp(weather_stress, 0, RISK_WEIGHTS["weather_stress"])

    # ── Factor 4: Market loss risk (0–10 pts) ────────────────────────────────
    modal     = float(market.get("modal_price", 1500))
    msp       = market.get("msp")
    trend     = market.get("trend", "stable").lower()

    market_score = 0
    if msp and modal < msp:
        market_score += 7
        reasons.append(
            f"Market price (₹{modal}/q) is BELOW MSP (₹{msp}/q) — consider government procurement."
        )
    if trend == "falling":
        market_score += 4
        reasons.append("Mandi prices are falling — selling soon may reduce further losses.")
    components["market_loss_risk"] = _clamp(market_score, 0, RISK_WEIGHTS["market_loss_risk"])

    # ── Factor 5: Rainfall deficit (0–10 pts) ────────────────────────────────
    rain_1h      = float(cur.get("rainfall_1h_mm", cur.get("rainfall", 0)))
    adv          = weather.get("farming_advisory", {})
    irr_needed   = adv.get("irrigation_needed", False)

    rain_score = 0
    if irr_needed or (rain_1h == 0 and humidity < 50):
        rain_score = 8
        reasons.append("Irrigation required — no rainfall and low humidity detected.")
    elif rain_1h == 0 and humidity < 65:
        rain_score = 4
        reasons.append("Rainfall below normal — monitor soil moisture carefully.")
    components["rainfall_deficit"] = _clamp(rain_score, 0, RISK_WEIGHTS["rainfall_deficit"])

    # ── Composite score ───────────────────────────────────────────────────────
    total = int(sum(components.values()))
    total = _clamp(total, 0, 100)

    if not reasons:
        reasons.append("All indicators are within safe ranges — continue normal farming practices.")

    return {
        "risk_score"      : total,
        "risk_level"      : _score_to_level(total),
        "component_scores": components,
        "risk_reasons"    : reasons,
        "is_healthy"      : is_healthy,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(language: str = "English") -> str:
    """
    The system prompt establishes the LLM's persona and output rules.
    Injects a language instruction so all advisory text is produced in the
    farmer's preferred language in a single LLM call — no post-translation needed.
    """
    lang_cfg = get_language_config(language)
    lang_instruction = lang_cfg["instruction"]

    return (
        "You are Kisaan Mitra (किसान मित्र / ಕಿಸಾನ್ ಮಿತ್ರ), an expert agricultural advisor "
        "for Indian smallholder farmers. You know Indian crop calendars, mandi markets, "
        "Kharif/Rabi seasons, government schemes (PM-KISAN, PMFBY, eNAM), and locally "
        "available inputs like DAP, Urea, neem oil, Bordeaux mixture, and Trichoderma.\n\n"
        f"LANGUAGE RULE: {lang_instruction}\n\n"
        "CRITICAL OUTPUT RULES:\n"
        "1. Respond ONLY with a valid JSON object — no prose before or after.\n"
        "2. Every field is required.\n"
        "3. Keep each body/step under 60 words — practical, not academic.\n"
        "4. Use ₹ for Indian Rupees. Mention specific product names (e.g. Mancozeb, DAP).\n"
        "5. Mention exact timings (e.g. 'within 24 hours', 'early morning').\n"
        "6. Never recommend illegal or banned pesticides in India."
    )


def _build_user_prompt(inputs: dict, risk: dict, language: str = "English") -> str:
    """
    Build the structured user prompt with all farm context.
    Uses explicit JSON output schema so the LLM knows exactly what to produce.
    """
    crop          = inputs.get("crop", "Unknown Crop")
    disease_label = inputs.get("disease_label", "None detected")
    confidence    = inputs.get("confidence", 0.0)
    weather       = inputs.get("weather") or {}
    market        = inputs.get("market")  or {}
    is_healthy    = risk.get("is_healthy", True)

    # ── Weather context block ─────────────────────────────────────────────────
    cur = weather.get("current", weather)
    if cur:
        weather_block = (
            f"Temperature: {cur.get('temp_c', cur.get('temp', 'N/A'))}°C, "
            f"Feels like: {cur.get('feels_like_c', cur.get('feels_like', 'N/A'))}°C, "
            f"Humidity: {cur.get('humidity_pct', cur.get('humidity', 'N/A'))}%, "
            f"Rainfall (last 1h): {cur.get('rainfall_1h_mm', cur.get('rainfall', 0))} mm, "
            f"Wind: {cur.get('wind_speed_kmh', 'N/A')} km/h, "
            f"Condition: {cur.get('condition', cur.get('description', 'N/A'))}"
        )
    else:
        weather_block = "Weather data not available."

    # ── Forecast context block ────────────────────────────────────────────────
    forecast = weather.get("forecast", [])
    if forecast:
        tomorrow = forecast[0]
        forecast_block = (
            f"Tomorrow: {tomorrow.get('condition', 'N/A')}, "
            f"High {tomorrow.get('temp_max_c', 'N/A')}°C, "
            f"Rain {tomorrow.get('rainfall_mm', 0)} mm "
            f"({int(tomorrow.get('rain_probability', 0) * 100)}% chance)"
        )
    else:
        forecast_block = "Forecast not available."

    # ── Farming advisory flags ────────────────────────────────────────────────
    adv = weather.get("farming_advisory", {})
    adv_flags = []
    if adv.get("irrigation_needed"):   adv_flags.append("IRRIGATION NEEDED")
    if adv.get("heat_stress_risk"):    adv_flags.append("HEAT STRESS RISK")
    if adv.get("frost_risk"):          adv_flags.append("FROST RISK")
    if adv.get("spray_conditions_ok"): adv_flags.append("SPRAY CONDITIONS GOOD")
    if adv.get("rain_expected_24h"):   adv_flags.append("RAIN EXPECTED IN 24H")
    if adv.get("high_humidity_disease_risk"): adv_flags.append("HIGH HUMIDITY — FUNGAL RISK")

    # ── Market context block ──────────────────────────────────────────────────
    msp = market.get("msp")
    msp_str = f"₹{msp}/quintal" if msp else "Not applicable"
    below_msp = msp and float(market.get("modal_price", 0)) < float(msp)

    market_block = (
        f"Modal: ₹{market.get('modal_price', 'N/A')}/quintal, "
        f"Range: ₹{market.get('min_price', 'N/A')}–₹{market.get('max_price', 'N/A')}/quintal, "
        f"Trend: {market.get('trend', 'unknown').upper()}, "
        f"MSP: {msp_str}, "
        f"Market: {market.get('market', 'Local Mandi')}"
    )

    # ── Disease context block ─────────────────────────────────────────────────
    conf_pct = f"{confidence*100:.1f}%"
    if is_healthy:
        disease_block = f"No disease detected — plant appears HEALTHY (confidence: {conf_pct})."
    else:
        disease_block = (
            f"Detected: {disease_label} (confidence: {conf_pct})\n"
            f"Severity: {_get_disease_severity(disease_label)}/100\n"
            f"Risk Level: {risk['risk_level']}"
        )

    # ── Full prompt ───────────────────────────────────────────────────────────
    lang_cfg  = get_language_config(language)
    json_note = lang_cfg["json_note"]
    prompt = f"""
Generate a complete farming advisory for this farmer's situation.

## FARMER CONTEXT
Crop          : {crop}
Disease       : {disease_block}
Weather       : {weather_block}
Forecast      : {forecast_block}
Active Flags  : {', '.join(adv_flags) if adv_flags else 'None'}
Market        : {market_block}
Below MSP?    : {'YES — Government procurement recommended' if below_msp else 'No'}
Risk Score    : {risk['risk_score']}/100 ({risk['risk_level']})
Risk Reasons  : {' | '.join(risk['risk_reasons'][:3])}

## LANGUAGE REQUIREMENT
{json_note}

## REQUIRED JSON OUTPUT FORMAT
Return EXACTLY this JSON structure (no markdown fences, no extra keys):

{{
  "summary": "<One sentence: what is the most urgent thing this farmer should do today>",

  "disease_warning": {{
    "heading": "<Short heading, max 8 words>",
    "body": "<What is this disease, is it serious, what happens if untreated. Max 55 words.>",
    "urgency": "<Act within X hours/days | Monitor closely | No action needed>"
  }},

  "irrigation_advice": {{
    "heading": "<Short heading>",
    "body": "<Exact irrigation schedule based on temp/humidity/rain data. Include frequency, timing of day, method (drip/furrow/flood). Max 50 words.>"
  }},

  "crop_inputs": {{
    "heading": "<Short heading>",
    "body": "<Specific fertilizer/pesticide/fungicide to apply. Include product name, dose, application method, timing. Max 60 words.>"
  }},

  "market_advice": {{
    "heading": "<Short heading>",
    "body": "<Should farmer sell now or hold? Why? Consider trend, MSP, price range. Include strategy. Max 50 words.>"
  }},

  "action_plan": {{
    "heading": "7-Day Action Plan",
    "steps": [
      "<Day 1–2: Specific action>",
      "<Day 2–3: Specific action>",
      "<Day 4–5: Specific action>",
      "<Day 6–7: Specific action>"
    ]
  }}
}}
""".strip()

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(
    system_prompt : str,
    user_prompt   : str,
    api_key       : str,
    provider      : str,
    session       : requests.Session,
) -> tuple:
    """
    Send messages to the LLM and return (raw_text, model_name, error_string).
    Parses the JSON response and returns the content string.
    """
    cfg = PROVIDER_CONFIGS[provider]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
    }

    payload = {
        "model"      : cfg["model"],
        "max_tokens" : cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "messages"   : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }

    try:
        resp = session.post(cfg["url"], headers=headers, json=payload, timeout=45)

        if resp.status_code == 401:
            return None, cfg["model"], (
                f"Invalid {provider.upper()} API key. "
                f"Check your {cfg['env_key']} in the .env file."
            )
        if resp.status_code == 429:
            return None, cfg["model"], (
                f"{provider.upper()} rate limit hit. Wait 60 seconds and try again."
            )

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content, cfg["model"], None

    except requests.exceptions.Timeout:
        return None, cfg["model"], "LLM request timed out (45s). Try again."
    except requests.exceptions.ConnectionError:
        return None, cfg["model"], "No internet connection. Check your network."
    except requests.exceptions.HTTPError as e:
        return None, cfg["model"], f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, cfg["model"], f"Unexpected error: {e}"


def _parse_llm_json(raw_text: str) -> tuple:
    """
    Parse the LLM's JSON response.
    Strips markdown code fences if present (LLMs sometimes add them despite instructions).
    Returns (parsed_dict, error_string).
    """
    if not raw_text:
        return None, "Empty response from LLM."

    # Strip ```json ... ``` fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError as e:
        # Try to find JSON object within the text (LLM sometimes adds preamble)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end])
                return parsed, None
            except json.JSONDecodeError:
                pass
        return None, f"Could not parse LLM response as JSON: {e}\nRaw: {raw_text[:300]}"


def _validate_sections(parsed: dict) -> dict:
    """
    Ensure all required keys exist in the parsed LLM response.
    Fills in placeholders for any missing section so the app never crashes.
    """
    defaults = {
        "summary": "Advisory generated — review sections below.",
        "disease_warning": {
            "heading": "Disease Status",
            "body"   : "Check the detected disease and apply appropriate treatment.",
            "urgency": "Consult local agricultural officer.",
        },
        "irrigation_advice": {
            "heading": "Irrigation",
            "body"   : "Maintain regular irrigation based on crop water requirements.",
        },
        "crop_inputs": {
            "heading": "Crop Inputs",
            "body"   : "Apply balanced fertilizer and consult local agro dealer for pesticide advice.",
        },
        "market_advice": {
            "heading": "Market Strategy",
            "body"   : "Monitor local mandi prices before selling. Compare with MSP.",
        },
        "action_plan": {
            "heading": "Action Plan",
            "steps"  : [
                "Inspect all plants for disease spread today.",
                "Apply recommended treatment within 48 hours.",
                "Monitor weather and irrigate as needed.",
                "Check mandi prices before harvest.",
            ],
        },
    }

    if not isinstance(parsed, dict):
        return defaults

    result = {}
    for key, default in defaults.items():
        if key in parsed and parsed[key]:
            result[key] = parsed[key]
        else:
            result[key] = default

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — MOCK ADVISORY (no API key / demo mode)
# ══════════════════════════════════════════════════════════════════════════════

def _build_mock_advisory(inputs: dict, risk: dict) -> dict:
    """
    Generate a high-quality mock advisory without any API call.
    Used when no API key is configured.
    """
    crop          = inputs.get("crop", "Tomato")
    disease_label = inputs.get("disease_label", "")
    is_healthy    = risk.get("is_healthy", True)
    weather       = inputs.get("weather") or {}
    market        = inputs.get("market")  or {}
    cur           = weather.get("current", weather)
    trend         = market.get("trend", "stable")

    # ── Disease warning ───────────────────────────────────────────────────────
    if is_healthy:
        disease_heading = "✅ Crop Looks Healthy"
        disease_body    = (
            f"No disease detected in your {crop} plant. Continue regular monitoring. "
            "Scout plants every 3–4 days and remove any yellowing or spotted leaves early."
        )
        disease_urgency = "No action needed — continue preventive care."
    else:
        sev = _get_disease_severity(disease_label)
        if sev >= 75:
            disease_heading = f"🚨 URGENT: {disease_label} Detected"
            disease_body    = (
                f"{disease_label} is a fast-spreading disease. If untreated, it can destroy "
                f"50–80% of your {crop} yield within 7–10 days. Apply fungicide immediately."
            )
            disease_urgency = "Act within 24 hours."
        else:
            disease_heading = f"⚠️ {disease_label} Detected"
            disease_body    = (
                f"{disease_label} causes yield loss if left untreated. Remove infected leaves "
                f"today and apply recommended fungicide to protect your {crop} crop."
            )
            disease_urgency = "Treat within 2–3 days."

    # ── Irrigation ────────────────────────────────────────────────────────────
    temp       = float(cur.get("temp_c",       cur.get("temp",     28)))
    humidity   = float(cur.get("humidity_pct", cur.get("humidity", 65)))
    rain_1h    = float(cur.get("rainfall_1h_mm", cur.get("rainfall", 0)))

    if rain_1h > 2:
        irr_body = (
            "Rainfall recorded — skip irrigation today. "
            "Resume watering tomorrow morning if soil feels dry 2–3 inches below surface."
        )
    elif temp > 35:
        irr_body = (
            f"High temperature ({temp}°C) detected. Irrigate in the early morning (6–8 AM) "
            "and again at sunset. Avoid midday watering — evaporation loss is very high."
        )
    elif humidity < 50:
        irr_body = (
            f"Low humidity ({humidity}%) — soil drying out fast. "
            "Water at the base every alternate day. Mulch with dry grass or straw to retain moisture."
        )
    else:
        irr_body = (
            f"Conditions are moderate (Temp: {temp}°C, Humidity: {humidity}%). "
            "Water every 3–4 days in the morning. Check soil moisture at 3-inch depth before irrigating."
        )

    # ── Crop inputs ────────────────────────────────────────────────────────────
    crop_lower = crop.lower()
    if "tomato" in crop_lower or "potato" in crop_lower:
        if not is_healthy:
            inputs_body = (
                "Apply Mancozeb 75% WP @ 2g/litre water as foliar spray. "
                "For bacterial diseases, use Copper Oxychloride 50% WP @ 3g/litre. "
                "Spray early morning when wind is calm. Repeat after 10 days."
            )
        else:
            inputs_body = (
                "Apply 19:19:19 NPK @ 5g/litre as foliar spray for balanced nutrition. "
                "Top-dress with Urea @ 100 kg/hectare at vegetative stage. "
                "Use neem oil spray @ 5ml/litre as preventive pest control."
            )
    elif "rice" in crop_lower:
        inputs_body = (
            "Apply Tricyclazole 75% WP @ 0.6g/litre for Blast disease. "
            "For Brown Spot, use Propiconazole 25% EC @ 1ml/litre. "
            "Top-dress with Potash (MOP) @ 50 kg/hectare to strengthen crop immunity."
        )
    elif "wheat" in crop_lower:
        inputs_body = (
            "Apply Propiconazole 25% EC @ 1ml/litre for rust control. "
            "Spray Tebuconazole if Yellow Rust is severe. "
            "Apply Zinc Sulphate @ 25 kg/hectare if leaves show yellowing between veins."
        )
    else:
        inputs_body = (
            "Consult your local Krishi Vigyan Kendra (KVK) for crop-specific input recommendations. "
            "As a general measure, apply neem-based products for pest control "
            "and ensure balanced NPK nutrition."
        )

    # ── Market advice ─────────────────────────────────────────────────────────
    modal  = market.get("modal_price", 1500)
    msp    = market.get("msp")
    market_name = market.get("market", "local mandi")

    if trend == "rising":
        market_body = (
            f"Prices at {market_name} are RISING (₹{modal}/quintal). "
            "Hold 30–40% of your produce for 1–2 weeks to benefit from the uptrend. "
            "Sell the rest now to manage cash flow."
        )
    elif trend == "falling":
        market_body = (
            f"Prices are FALLING (₹{modal}/quintal). "
            f"Sell your {crop} within the next 3–5 days to avoid further losses. "
            + (f"If price drops below MSP (₹{msp}/q), approach your nearest APMC or FCI." if msp else "")
        )
    else:
        market_body = (
            f"Prices are STABLE at ₹{modal}/quintal. "
            "No urgency to sell — maintain quality through proper storage. "
            "Monitor Agmarknet (agmarknet.gov.in) daily for price updates."
        )

    # ── Action plan ────────────────────────────────────────────────────────────
    steps = []
    if not is_healthy:
        steps.append(f"Day 1: Apply fungicide/bactericide spray on all {crop} plants this morning.")
        steps.append("Day 2–3: Remove and destroy all infected leaves and stems.")
    else:
        steps.append(f"Day 1–2: Scout all {crop} plants for early signs of disease or pests.")

    steps.append(f"Day 3–4: Check soil moisture and {'resume' if rain_1h > 0 else 'continue'} irrigation schedule.")
    steps.append("Day 5: Apply foliar fertilizer (19:19:19 NPK) in the morning.")

    if trend in ("rising",):
        steps.append("Day 6–7: Check mandi prices — if holding, ensure proper storage and grading.")
    else:
        steps.append("Day 6–7: Prepare crop for sale — grade, weigh, and transport to nearest mandi.")

    return {
        "summary"          : f"{'Immediate disease treatment required for' if not is_healthy else 'Monitor your healthy'} {crop} — {disease_urgency if not is_healthy else 'continue regular care.'}",
        "disease_warning"  : {"heading": disease_heading, "body": disease_body, "urgency": disease_urgency},
        "irrigation_advice": {"heading": "💧 Irrigation Schedule",       "body": irr_body},
        "crop_inputs"      : {"heading": "🌱 Pesticide & Fertilizer",    "body": inputs_body},
        "market_advice"    : {"heading": "📊 Mandi Market Strategy",     "body": market_body},
        "action_plan"      : {"heading": "📋 7-Day Action Plan",         "steps": steps},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC API — generate_advisory()
# ══════════════════════════════════════════════════════════════════════════════

def generate_advisory(
    inputs   : dict,
    api_key  : str  = None,
    provider : str  = DEFAULT_PROVIDER,
    language : str  = DEFAULT_LANGUAGE,
) -> dict:
    """
    Generate a complete, structured agricultural advisory.

    This is the main public function. Call this from app.py.

    Args:
        inputs: dict with the following keys (all optional — graceful fallbacks):
            crop          (str)   — crop name, e.g. "Tomato"
            disease_label (str)   — e.g. "Tomato - Early Blight" or "Tomato - Healthy"
            confidence    (float) — model confidence 0.0–1.0
            weather       (dict)  — output of weather_fetcher.get_weather()
            market        (dict)  — output of market_prices.get_market_prices()

        api_key  : Groq or OpenAI API key. If None, reads from environment.
        provider : "groq" (default) or "openai"

    Returns:
        AdvisoryResult dict with keys:
            risk_level, risk_score, risk_reasons,
            sections (disease_warning, irrigation_advice, crop_inputs,
                      market_advice, action_plan),
            summary, source, model_used, generated_at, error

    Example:
        >>> from utils.advisor import generate_advisory
        >>> result = generate_advisory(
        ...     inputs={
        ...         "crop"         : "Tomato",
        ...         "disease_label": "Tomato - Early Blight",
        ...         "confidence"   : 0.91,
        ...         "weather"      : weather_data,   # from get_weather()
        ...         "market"       : market_data,    # from get_market_prices()
        ...     },
        ...     api_key=os.environ["GROQ_API_KEY"],
        ...     provider="groq",
        ... )
        >>> print(result["risk_level"])
        'HIGH'
        >>> print(result["sections"]["disease_warning"]["urgency"])
        'Act within 24 hours.'
    """
    if provider not in PROVIDER_CONFIGS:
        provider = DEFAULT_PROVIDER

    cfg = PROVIDER_CONFIGS[provider]

    # ── Resolve API key ────────────────────────────────────────────────────────
    if not api_key:
        api_key = os.environ.get(cfg["env_key"], "")

    no_key = (not api_key or api_key.strip() in ("", cfg["placeholder"]))

    # ── Step 1: Risk scoring (always runs, no API needed) ──────────────────────
    risk = compute_risk_score(inputs)

    # ── Step 2: Mock mode ─────────────────────────────────────────────────────
    if no_key:
        print(f"[Advisor] No {provider.upper()} API key found. Using mock advisory.")
        sections = _build_mock_advisory(inputs, risk)
        return {
            "risk_level"      : risk["risk_level"],
            "risk_score"      : risk["risk_score"],
            "component_scores": risk["component_scores"],
            "risk_reasons"    : risk["risk_reasons"],
            "sections"        : sections,
            "summary"         : sections["summary"],
            "source"          : "mock",
            "model_used"      : "rule-based",
            "generated_at"    : _ist_now(),
            "language"        : language,
            "error"           : None,
            "_mock"           : True,
        }

    # ── Step 3: Build prompts ─────────────────────────────────────────────────
    system_prompt = _build_system_prompt(language)
    user_prompt   = _build_user_prompt(inputs, risk, language)

    # ── Step 4: Call LLM ──────────────────────────────────────────────────────
    session = _make_session()
    t0 = time.time()

    print(f"[Advisor] Calling {provider.upper()} ({cfg['model']}) ...")
    raw_text, model_name, llm_error = _call_llm(
        system_prompt, user_prompt, api_key, provider, session
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"[Advisor] LLM responded in {elapsed_ms}ms")

    if llm_error:
        # Fall back to mock advisory on LLM failure so app never crashes
        print(f"[Advisor] LLM error: {llm_error} — falling back to mock.")
        sections = _build_mock_advisory(inputs, risk)
        return {
            "risk_level"      : risk["risk_level"],
            "risk_score"      : risk["risk_score"],
            "component_scores": risk["component_scores"],
            "risk_reasons"    : risk["risk_reasons"],
            "sections"        : sections,
            "summary"         : sections["summary"],
            "source"          : "mock_fallback",
            "model_used"      : model_name,
            "generated_at"    : _ist_now(),
            "language"        : language,
            "error"           : llm_error,
            "_mock"           : True,
        }

    # ── Step 5: Parse JSON response ────────────────────────────────────────────
    parsed, parse_error = _parse_llm_json(raw_text)

    if parse_error:
        print(f"[Advisor] JSON parse error: {parse_error}")
        # Still return what we have — use mock for sections, real risk score
        sections = _build_mock_advisory(inputs, risk)
        sections["summary"] = parsed.get("summary", sections["summary"]) if parsed else sections["summary"]
        return {
            "risk_level"      : risk["risk_level"],
            "risk_score"      : risk["risk_score"],
            "component_scores": risk["component_scores"],
            "risk_reasons"    : risk["risk_reasons"],
            "sections"        : sections,
            "summary"         : sections["summary"],
            "source"          : provider,
            "model_used"      : model_name,
            "generated_at"    : _ist_now(),
            "error"           : f"JSON parse error: {parse_error}",
        }

    # ── Step 6: Validate and fill missing sections ─────────────────────────────
    sections = _validate_sections(parsed)

    return {
        "risk_level"      : risk["risk_level"],
        "risk_score"      : risk["risk_score"],
        "component_scores": risk["component_scores"],
        "risk_reasons"    : risk["risk_reasons"],
        "sections"        : sections,
        "summary"         : sections.get("summary", parsed.get("summary", "")),
        "source"          : provider,
        "model_used"      : model_name,
        "generated_at"    : _ist_now(),
        "inference_ms"    : elapsed_ms,
        "language"        : language,
        "error"           : None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def format_advisory_markdown(result: dict) -> str:
    """
    Convert an AdvisoryResult dict into a clean markdown string.
    Useful for Streamlit st.markdown() or saving to a file.

    Args:
        result: Output of generate_advisory()

    Returns:
        Markdown-formatted string of the full advisory.
    """
    if result.get("error") and not result.get("sections"):
        return f"❌ **Advisory Error:** {result['error']}"

    lines = []

    # Mock badge
    if result.get("_mock"):
        lines.append("> ⚠️ **Demo Mode** — Add your Groq API key in `.env` for AI-generated advice.\n")

    # Summary
    summary = result.get("summary", "")
    if summary:
        lines.append(f"### 📌 Summary\n{summary}\n")

    # Risk badge
    risk_level = result.get("risk_level", "NONE")
    risk_score = result.get("risk_score", 0)
    risk_emoji = {"NONE": "🟢", "LOW": "🟡", "MODERATE": "🟠", "HIGH": "🔴", "CRITICAL": "⛔"}.get(risk_level, "⚪")
    lines.append(f"**{risk_emoji} Risk Level: {risk_level}** &nbsp;&nbsp; *(Score: {risk_score}/100)*\n")

    # Risk reasons
    reasons = result.get("risk_reasons", [])
    if reasons:
        lines.append("**Why:**")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("---\n")

    # Advisory sections
    sections = result.get("sections", {})
    order    = ["disease_warning", "irrigation_advice", "crop_inputs", "market_advice", "action_plan"]

    for key in order:
        section = sections.get(key, {})
        if not section:
            continue

        emoji   = SECTION_EMOJIS.get(key, "📌")
        heading = section.get("heading", key.replace("_", " ").title())
        lines.append(f"### {emoji} {heading}")

        if key == "action_plan":
            steps = section.get("steps", [])
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
        else:
            body = section.get("body", "")
            if body:
                lines.append(body)
            urgency = section.get("urgency", "")
            if urgency:
                lines.append(f"\n**⏱ Urgency:** {urgency}")

        lines.append("")

    # Footer
    source = result.get("source", "unknown")
    model  = result.get("model_used", "")
    ts     = result.get("generated_at", "")
    lines.append(f"---\n*Generated by {model} ({source}) · {ts}*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION HELPER — translate an existing English advisory result
# ══════════════════════════════════════════════════════════════════════════════

def translate_advisory(
    result   : dict,
    language : str,
    api_key  : str  = None,
    provider : str  = DEFAULT_PROVIDER,
) -> dict:
    """
    Translate a previously generated English advisory into Hindi or Kannada.

    Use this when the user switches language AFTER an advisory has already been
    generated — avoids re-running the full LLM advisory pipeline.

    The LLM receives the five text sections and produces translated versions,
    keeping the JSON structure identical. Numbers, ₹ amounts, product names,
    and dosages are preserved exactly.

    Args:
        result   : Output dict from generate_advisory() (must have "sections")
        language : Target language — "Hindi" or "Kannada" ("English" is a no-op)
        api_key  : LLM API key (reads from env if None)
        provider : "groq" or "openai"

    Returns:
        New result dict with translated sections, same schema as generate_advisory().
        On failure, returns the original result unchanged with an error note.
    """
    if language == "English" or language == result.get("language", "English"):
        # Nothing to translate — return as-is
        return {**result, "language": language}

    if provider not in PROVIDER_CONFIGS:
        provider = DEFAULT_PROVIDER
    cfg = PROVIDER_CONFIGS[provider]

    if not api_key:
        api_key = os.environ.get(cfg["env_key"], "")
    no_key = (not api_key or api_key.strip() in ("", cfg["placeholder"]))

    if no_key:
        # No API key — return original with language tag changed
        return {**result, "language": language, "_translation_note": "No API key — translation skipped"}

    lang_cfg  = get_language_config(language)
    sections  = result.get("sections", {})

    # Build a compact translation payload — only the text fields
    text_payload = {
        "summary"  : result.get("summary", ""),
        "sections" : {
            key: {
                k: v for k, v in sec.items()
                if isinstance(v, (str, list))   # text fields only; skip non-strings
            }
            for key, sec in sections.items()
        },
    }

    system_msg = (
        f"You are a professional agricultural translator. "
        f"{lang_cfg['instruction']} "
        "Translate the provided JSON farming advisory text. "
        "Rules: "
        "1. Return ONLY valid JSON — no prose, no markdown fences. "
        "2. Keep the EXACT same JSON structure and keys. "
        "3. Keep all numbers, ₹ amounts, percentages, and product names (Mancozeb, DAP, Urea) unchanged. "
        "4. Translate ALL string values (including list items in 'steps'). "
        "5. Do NOT add or remove any keys."
    )

    user_msg = (
        f"Translate this farming advisory JSON into {language}. "
        f"{lang_cfg['json_note']}\n\n"
        f"Input JSON:\n{json.dumps(text_payload, ensure_ascii=False, indent=2)}"
    )

    session = _make_session()
    raw_text, model_name, llm_error = _call_llm(system_msg, user_msg, api_key, provider, session)

    if llm_error:
        return {**result, "language": language,
                "_translation_note": f"Translation failed: {llm_error}"}

    parsed, parse_error = _parse_llm_json(raw_text)
    if parse_error or not isinstance(parsed, dict):
        return {**result, "language": language,
                "_translation_note": f"Translation parse error: {parse_error}"}

    # Merge translated text back into full result
    translated_sections = {}
    for key, sec in sections.items():
        translated_sec = parsed.get("sections", {}).get(key, {})
        if translated_sec:
            # Layer translated text over original structure (preserves any extra keys)
            translated_sections[key] = {**sec, **translated_sec}
        else:
            translated_sections[key] = sec   # Fallback: keep original section

    return {
        **result,
        "summary"  : parsed.get("summary", result.get("summary", "")),
        "sections" : translated_sections,
        "language" : language,
        "_translated_by": model_name,
    }


# Backward-compatible wrapper for app.py calls
def generate_recommendation(
    crop          : str,
    disease_result: Optional[dict],
    weather       : Optional[dict],
    market        : Optional[dict],
    api_key       : str,
    provider      : str = DEFAULT_PROVIDER,
    language      : str = DEFAULT_LANGUAGE,
) -> str:
    """
    Backward-compatible wrapper for the old app.py call signature.
    Returns markdown string (same as before).
    """
    inputs = {
        "crop"         : crop,
        "disease_label": (disease_result or {}).get("label", ""),
        "confidence"   : (disease_result or {}).get("confidence", 0.0),
        "weather"      : weather,
        "market"       : market,
    }
    result = generate_advisory(inputs, api_key=api_key, provider=provider, language=language)
    return format_advisory_markdown(result)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_risk_only(risk: dict):
    W = 58
    print("\n" + "="*W)
    print("  🌾  Agri-Edge — Risk Assessment")
    print("="*W)
    lvl   = risk["risk_level"]
    score = risk["risk_score"]
    emoji = {"NONE": "🟢", "LOW": "🟡", "MODERATE": "🟠", "HIGH": "🔴", "CRITICAL": "⛔"}.get(lvl, "⚪")
    print(f"\n  {emoji}  Risk Level : {lvl}  ({score}/100)")
    print(f"\n  Component Breakdown:")
    for factor, pts in risk["component_scores"].items():
        bar = "█" * pts + "░" * (RISK_WEIGHTS.get(factor, 10) - pts)
        print(f"    {factor:<25}  [{bar}]  {pts} pts")
    print(f"\n  Risk Reasons:")
    for r in risk["risk_reasons"]:
        print(f"    • {r}")
    print("="*W + "\n")


def _print_advisory_human(result: dict):
    """Print advisory to terminal in a clean readable format."""
    W = 62
    print("\n" + "="*W)
    print("  🌾  Agri-Edge — Agricultural Advisory Report")
    print("="*W)

    if result.get("_mock"):
        print("  ⚠️  DEMO MODE — Add API key for AI-generated advice")

    risk_level = result.get("risk_level", "NONE")
    risk_score = result.get("risk_score", 0)
    emoji = {"NONE": "🟢", "LOW": "🟡", "MODERATE": "🟠", "HIGH": "🔴", "CRITICAL": "⛔"}.get(risk_level, "⚪")

    print(f"\n  📌 {result.get('summary', '')}")
    print(f"\n  {emoji} Risk: {risk_level} ({risk_score}/100)")
    for r in result.get("risk_reasons", []):
        print(f"     • {r}")

    sections = result.get("sections", {})
    order    = ["disease_warning", "irrigation_advice", "crop_inputs", "market_advice", "action_plan"]
    emojis   = {"disease_warning": "🦠", "irrigation_advice": "💧",
                 "crop_inputs": "🌱", "market_advice": "📊", "action_plan": "📋"}

    for key in order:
        sec = sections.get(key)
        if not sec:
            continue
        print(f"\n  {emojis.get(key, '📌')} {sec.get('heading', key)}")
        print(f"  {'─' * 56}")
        if key == "action_plan":
            for i, step in enumerate(sec.get("steps", []), 1):
                print(f"    {i}. {step}")
        else:
            print(f"    {sec.get('body', '')}")
            if sec.get("urgency"):
                print(f"\n    ⏱ {sec['urgency']}")

    print(f"\n  {'─'*56}")
    print(f"  Source: {result.get('model_used')} ({result.get('source')})")
    print(f"  {result.get('generated_at', '')}")
    print("="*W + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Agri-Edge: Generate AI advisory from crop, disease, and weather data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With disease and PIN code (uses mock weather if no OpenWeather key):
  python utils/advisor.py --crop Tomato --disease "Tomato - Early Blight" --confidence 0.91 --pin 411001

  # Healthy crop, city name:
  python utils/advisor.py --crop Rice --no-disease --city Hyderabad

  # JSON output (pipe to other tools):
  python utils/advisor.py --crop Potato --disease "Potato - Late Blight" --confidence 0.95 --json

  # Risk score only (instant, no API call):
  python utils/advisor.py --risk-only --crop Tomato --disease "Tomato - Early Blight" --confidence 0.87

  # Use OpenAI instead of Groq:
  python utils/advisor.py --crop Wheat --disease "Wheat - Yellow Rust" --confidence 0.78 --provider openai
        """,
    )

    parser.add_argument("--crop",       "-c",  type=str, default="Tomato",
                        help="Crop name (default: Tomato)")
    parser.add_argument("--disease",    "-d",  type=str, default="",
                        help="Detected disease label, e.g. 'Tomato - Early Blight'")
    parser.add_argument("--no-disease", "-nd", action="store_true",
                        help="Indicate healthy plant (no disease)")
    parser.add_argument("--confidence", "-cf", type=float, default=0.80,
                        help="Model confidence 0.0–1.0 (default: 0.80)")
    parser.add_argument("--pin",        "-p",  type=str, default=None,
                        help="Indian PIN code for weather lookup")
    parser.add_argument("--city",       "-l",  type=str, default=None,
                        help="City name for weather lookup (alternative to --pin)")
    parser.add_argument("--provider",         type=str, default=DEFAULT_PROVIDER,
                        choices=["groq", "openai"],
                        help="LLM provider (default: groq)")
    parser.add_argument("--key",        "-k",  type=str, default=None,
                        help="LLM API key (overrides env var)")
    parser.add_argument("--json",       "-j",  action="store_true",
                        help="Output raw JSON")
    parser.add_argument("--risk-only",  "-r",  action="store_true",
                        help="Only run risk scoring (no LLM call, instant)")

    args = parser.parse_args()

    # ── Resolve disease label ─────────────────────────────────────────────────
    if args.no_disease:
        disease_label = f"{args.crop} - Healthy"
    elif args.disease:
        disease_label = args.disease
    else:
        disease_label = f"{args.crop} - Healthy"

    # ── Fetch weather if location given ───────────────────────────────────────
    weather_data = {}
    if args.pin or args.city:
        try:
            from utils.weather_fetcher import get_weather
            ow_key  = os.environ.get("OPENWEATHER_API_KEY", "")
            location = args.pin or args.city
            print(f"[CLI] Fetching weather for: {location}")
            weather_data = get_weather(location, ow_key)
        except ImportError:
            print("[CLI] weather_fetcher not available — proceeding without weather data.")

    # ── Get market prices ─────────────────────────────────────────────────────
    try:
        from utils.market_prices import get_market_prices
        market_data = get_market_prices(args.crop)
    except ImportError:
        market_data = {}

    # ── Build inputs ──────────────────────────────────────────────────────────
    inputs = {
        "crop"         : args.crop,
        "disease_label": disease_label,
        "confidence"   : args.confidence,
        "weather"      : weather_data,
        "market"       : market_data,
    }

    # ── Risk only mode ────────────────────────────────────────────────────────
    if args.risk_only:
        risk = compute_risk_score(inputs)
        if args.json:
            print(json.dumps(risk, indent=2))
        else:
            _print_risk_only(risk)
        return

    # ── Full advisory ─────────────────────────────────────────────────────────
    result = generate_advisory(inputs, api_key=args.key, provider=args.provider)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_advisory_human(result)

    if result.get("error") and not result.get("_mock"):
        sys.exit(1)


if __name__ == "__main__":
    main()
