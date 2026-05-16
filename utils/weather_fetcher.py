"""
utils/weather_fetcher.py
========================
Hyperlocal Weather Data — OpenWeatherMap API (India / PIN Code)
---------------------------------------------------------------
Fetches current conditions + 5-day / 3-hourly forecast for any
Indian location, identified by PIN code or city name.

PUBLIC FUNCTIONS
----------------
    get_weather_by_pin(pin_code, api_key)   → full weather dict
    get_weather_by_city(city, api_key)      → full weather dict
    get_weather(location, api_key)          → auto-detects PIN or city name
    get_forecast(location, api_key)         → 5-day forecast list

RETURN SCHEMA (all functions)
------------------------------
{
  "location": {
      "city"        : "Nashik",
      "state"       : "Maharashtra",
      "country"     : "IN",
      "pin_code"    : "422001",
      "lat"         : 19.9975,
      "lon"         : 73.7898,
  },
  "current": {
      "temp_c"          : 31.4,   # Celsius
      "feels_like_c"    : 34.2,
      "temp_min_c"      : 28.1,
      "temp_max_c"      : 33.6,
      "humidity_pct"    : 72,
      "pressure_hpa"    : 1009,
      "wind_speed_kmh"  : 13.0,
      "wind_direction"  : "NW",
      "visibility_km"   : 10.0,
      "cloud_cover_pct" : 40,
      "condition"       : "Partly cloudy",
      "condition_code"  : 802,
      "icon_code"       : "02d",
      "uv_index"        : None,  # Requires One Call API (paid)
      "rainfall_1h_mm"  : 0.0,
      "rainfall_3h_mm"  : 0.0,
  },
  "forecast": [            # Next 5 days (daily summary)
      {
          "date"            : "2024-06-15",
          "day_name"        : "Saturday",
          "temp_min_c"      : 26.0,
          "temp_max_c"      : 34.5,
          "humidity_pct"    : 68,
          "rainfall_mm"     : 2.5,
          "rain_probability": 0.4,
          "condition"       : "Light rain",
          "icon_code"       : "10d",
      },
      ...
  ],
  "farming_advisory": {
      "irrigation_needed"     : False,
      "frost_risk"            : False,
      "heat_stress_risk"      : True,
      "rain_expected_24h"     : True,
      "spray_conditions_ok"   : False,   # Wind < 15 km/h and no rain
      "advisory_notes"        : ["High humidity — watch for fungal disease.", ...],
  },
  "fetched_at"  : "2024-06-14 08:30:00 IST",
  "source"      : "openweathermap",   # or "mock"
  "error"       : None,
}

API KEY SETUP
-------------
1. Sign up free at https://openweathermap.org/api
2. Navigate to: API keys → Generate a new key
3. Wait ~10 minutes for the key to activate
4. Add to your .env file:
       OPENWEATHER_API_KEY=abc123yourkeyhere
   Or pass directly:
       get_weather("411001", api_key="abc123yourkeyhere")

FREE TIER LIMITS
----------------
  - Current weather : 1,000 calls/day  (endpoint: /data/2.5/weather)
  - 5-day forecast  : 1,000 calls/day  (endpoint: /data/2.5/forecast)
  - PIN code lookup : uses /zip endpoint — same quota as current weather

CLI USAGE
---------
    python utils/weather_fetcher.py --pin 411001
    python utils/weather_fetcher.py --city Nashik
    python utils/weather_fetcher.py --pin 600001 --json
    python utils/weather_fetcher.py --pin 110001 --forecast
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime, timezone, timedelta
from typing import Union, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# OpenWeatherMap API endpoints (free tier)
BASE_URL_CURRENT  = "https://api.openweathermap.org/data/2.5/weather"
BASE_URL_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"
BASE_URL_GEO      = "https://api.openweathermap.org/geo/1.0/zip"

# India Standard Time offset: UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

# Agri-relevant thresholds (metric units)
HEAT_STRESS_THRESHOLD_C  = 35.0   # Above this → crop heat stress warning
FROST_RISK_THRESHOLD_C   = 4.0    # Below this → frost risk alert
HIGH_WIND_THRESHOLD_KMH  = 20.0   # Above this → spraying not recommended
HIGH_HUMIDITY_THRESHOLD  = 80     # Above this → fungal disease risk
LOW_HUMIDITY_THRESHOLD   = 30     # Below this → irrigation urgency high
DROUGHT_RAIN_THRESHOLD   = 0.5    # mm/day below this → irrigation recommended

# HTTP request settings
REQUEST_TIMEOUT_SEC = 10
MAX_RETRIES         = 2           # Retry on connection errors / 5xx

# Placeholder used in settings.py / .env
_PLACEHOLDER_KEY = "your_openweather_api_key_here"

# Indian PIN code: exactly 6 digits, starts with 1–9
_PIN_PATTERN = re.compile(r"^[1-9]\d{5}$")

# Wind direction lookup (meteorological degrees → compass)
_WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _make_session() -> requests.Session:
    """
    Build a requests.Session with:
    - Automatic retry on connection errors and server errors (502, 503, 504)
    - 10-second timeout enforced at the call site
    """
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,                       # 0.5s, 1s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _degrees_to_compass(degrees: float) -> str:
    """Convert wind direction in meteorological degrees to compass label."""
    idx = round(degrees / 22.5) % 16
    return _WIND_DIRECTIONS[idx]


def _mps_to_kmh(mps: float) -> float:
    """Convert metres per second → kilometres per hour."""
    return round(mps * 3.6, 1)


def _ist_now_str() -> str:
    """Return current IST datetime as a human-readable string."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _is_valid_pin(pin: str) -> bool:
    """Return True if the string looks like a valid Indian PIN code."""
    return bool(_PIN_PATTERN.match(str(pin).strip()))


def _error_response(message: str, pin_code: str = None, city: str = None) -> dict:
    """Build a standardised error response dict."""
    return {
        "location"         : {"pin_code": pin_code, "city": city},
        "current"          : {},
        "forecast"         : [],
        "farming_advisory" : {},
        "fetched_at"       : _ist_now_str(),
        "source"           : "error",
        "error"            : message,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PIN CODE → COORDINATES (Geocoding)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_pin_to_coords(pin_code: str, api_key: str, session: requests.Session) -> dict:
    """
    Convert an Indian PIN code to lat/lon using OpenWeatherMap's Geocoding API.

    OWM accepts ZIP codes in the format "PINCODE,IN" for India.

    Returns:
        dict with keys: lat, lon, city, country
        or raises ValueError on failure.
    """
    params = {
        "zip"   : f"{pin_code},IN",
        "appid" : api_key,
    }

    try:
        resp = session.get(BASE_URL_GEO, params=params, timeout=REQUEST_TIMEOUT_SEC)

        # 404 means PIN not found in OWM database
        if resp.status_code == 404:
            raise ValueError(
                f"PIN code '{pin_code}' not found in OpenWeatherMap database.\n"
                f"Try using the city name instead (e.g. 'Nashik', 'Pune')."
            )

        resp.raise_for_status()
        data = resp.json()

        return {
            "lat"    : data["lat"],
            "lon"    : data["lon"],
            "city"   : data.get("name", "Unknown"),
            "country": data.get("country", "IN"),
        }

    except ValueError:
        raise
    except requests.exceptions.HTTPError as e:
        # Handle 401 Unauthorized (invalid API key)
        if e.response.status_code == 401:
            raise ValueError(
                "Invalid API key. Please check your OPENWEATHER_API_KEY.\n"
                "New keys take ~10 minutes to activate after creation."
            )
        raise ValueError(f"Geocoding failed (HTTP {e.response.status_code}): {e.response.text}")
    except requests.exceptions.ConnectionError:
        raise ValueError("Network error during geocoding. Check your internet connection.")
    except requests.exceptions.Timeout:
        raise ValueError("Geocoding request timed out. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE CURRENT WEATHER RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_current(data: dict, pin_code: str = None) -> dict:
    """
    Parse the raw OpenWeatherMap /data/2.5/weather JSON response
    into our clean structured format.
    """
    main    = data.get("main", {})
    wind    = data.get("wind", {})
    rain    = data.get("rain", {})
    clouds  = data.get("clouds", {})
    weather = data.get("weather", [{}])[0]
    coord   = data.get("coord", {})
    sys_    = data.get("sys", {})

    wind_speed_mps = wind.get("speed", 0.0)
    wind_deg       = wind.get("deg", 0)

    return {
        "location": {
            "city"      : data.get("name", "Unknown"),
            "state"     : "",          # Not returned by this endpoint
            "country"   : sys_.get("country", "IN"),
            "pin_code"  : pin_code or "",
            "lat"       : round(coord.get("lat", 0.0), 4),
            "lon"       : round(coord.get("lon", 0.0), 4),
        },
        "current": {
            "temp_c"          : round(main.get("temp", 0.0), 1),
            "feels_like_c"    : round(main.get("feels_like", 0.0), 1),
            "temp_min_c"      : round(main.get("temp_min", 0.0), 1),
            "temp_max_c"      : round(main.get("temp_max", 0.0), 1),
            "humidity_pct"    : main.get("humidity", 0),
            "pressure_hpa"    : main.get("pressure", 0),
            "wind_speed_kmh"  : _mps_to_kmh(wind_speed_mps),
            "wind_direction"  : _degrees_to_compass(wind_deg),
            "visibility_km"   : round(data.get("visibility", 0) / 1000, 1),
            "cloud_cover_pct" : clouds.get("all", 0),
            "condition"       : weather.get("description", "").capitalize(),
            "condition_code"  : weather.get("id", 0),
            "icon_code"       : weather.get("icon", "01d"),
            "uv_index"        : None,   # Requires paid One Call API
            "rainfall_1h_mm"  : round(rain.get("1h", 0.0), 2),
            "rainfall_3h_mm"  : round(rain.get("3h", 0.0), 2),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE FORECAST RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_forecast(data: dict) -> list:
    """
    Parse the raw /data/2.5/forecast JSON (3-hourly, 5 days).
    Aggregates into daily summaries: min/max temp, total rain, conditions.

    Returns a list of up to 5 daily dicts.
    """
    from collections import defaultdict

    daily = defaultdict(lambda: {
        "temps"       : [],
        "humidity"    : [],
        "rainfall_mm" : 0.0,
        "rain_probs"  : [],
        "conditions"  : [],
        "icons"       : [],
    })

    for slot in data.get("list", []):
        # Convert Unix timestamp → IST date string
        dt_utc = datetime.fromtimestamp(slot["dt"], tz=timezone.utc)
        dt_ist = dt_utc.astimezone(IST)
        date_str = dt_ist.strftime("%Y-%m-%d")

        main    = slot.get("main", {})
        rain    = slot.get("rain", {})
        weather = slot.get("weather", [{}])[0]

        daily[date_str]["temps"].append(main.get("temp", 0.0))
        daily[date_str]["humidity"].append(main.get("humidity", 0))
        daily[date_str]["rainfall_mm"] += rain.get("3h", 0.0)
        daily[date_str]["rain_probs"].append(slot.get("pop", 0.0))
        daily[date_str]["conditions"].append(weather.get("description", ""))
        daily[date_str]["icons"].append(weather.get("icon", "01d"))

    result = []
    # Skip today (index 0) and show the next 5 days
    sorted_dates = sorted(daily.keys())
    for date_str in sorted_dates[:5]:
        d       = daily[date_str]
        dt_obj  = datetime.strptime(date_str, "%Y-%m-%d")

        # Most common condition for the day
        condition = max(set(d["conditions"]), key=d["conditions"].count) if d["conditions"] else ""

        # Most common non-night icon (prefer "d" suffix for day icons)
        day_icons = [ic for ic in d["icons"] if ic.endswith("d")] or d["icons"]
        icon = max(set(day_icons), key=day_icons.count) if day_icons else "01d"

        result.append({
            "date"             : date_str,
            "day_name"         : dt_obj.strftime("%A"),
            "temp_min_c"       : round(min(d["temps"]), 1),
            "temp_max_c"       : round(max(d["temps"]), 1),
            "humidity_pct"     : round(sum(d["humidity"]) / len(d["humidity"])),
            "rainfall_mm"      : round(d["rainfall_mm"], 2),
            "rain_probability" : round(max(d["rain_probs"]), 2),
            "condition"        : condition.capitalize(),
            "icon_code"        : icon,
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FARMING ADVISORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _build_farming_advisory(current: dict, forecast: list) -> dict:
    """
    Derive simple rule-based farming advisories from weather data.
    These supplement the LLM advisory in advisor.py with concrete flags.

    Rules are deliberately simple — they should be auditable and explainable
    to a farmer without any AI knowledge.
    """
    temp        = current.get("temp_c", 25.0)
    humidity    = current.get("humidity_pct", 60)
    wind_kmh    = current.get("wind_speed_kmh", 0.0)
    rain_1h     = current.get("rainfall_1h_mm", 0.0)
    condition   = current.get("condition", "").lower()

    # Pull next-24h forecast data (first ~2 slots cover ~6 hours each)
    forecast_24h = forecast[:2] if forecast else []
    rain_24h     = sum(d.get("rainfall_mm", 0.0) for d in forecast_24h)
    rain_prob_24h= max((d.get("rain_probability", 0.0) for d in forecast_24h), default=0.0)

    # ── Evaluate flags ────────────────────────────────────────────────────────
    heat_stress      = temp >= HEAT_STRESS_THRESHOLD_C
    frost_risk       = temp <= FROST_RISK_THRESHOLD_C
    high_humidity    = humidity >= HIGH_HUMIDITY_THRESHOLD
    low_humidity     = humidity <= LOW_HUMIDITY_THRESHOLD
    high_wind        = wind_kmh >= HIGH_WIND_THRESHOLD_KMH
    raining_now      = rain_1h > 0 or "rain" in condition
    rain_expected    = rain_prob_24h >= 0.4 or rain_24h >= 1.0
    irrigation_needed= (not raining_now and not rain_expected and
                        rain_24h < DROUGHT_RAIN_THRESHOLD)
    spray_ok         = not raining_now and wind_kmh < HIGH_WIND_THRESHOLD_KMH

    # ── Build advisory notes ──────────────────────────────────────────────────
    notes = []

    if frost_risk:
        notes.append("🥶 FROST RISK: Temperature near or below 4°C — cover sensitive crops overnight.")

    if heat_stress:
        notes.append(
            f"🌡 HEAT STRESS: Temperature {temp}°C exceeds {HEAT_STRESS_THRESHOLD_C}°C — "
            "irrigate in the early morning or evening to reduce plant stress."
        )

    if high_humidity and not raining_now:
        notes.append(
            f"💧 HIGH HUMIDITY ({humidity}%): Conditions favour fungal diseases "
            "(blight, mildew, rust). Inspect crops and apply preventive fungicide."
        )

    if low_humidity:
        notes.append(
            f"☀️ LOW HUMIDITY ({humidity}%): Plants losing moisture rapidly. "
            "Increase irrigation frequency and consider mulching."
        )

    if irrigation_needed:
        notes.append(
            "🚿 IRRIGATION RECOMMENDED: No significant rainfall expected in the next 24 hours. "
            "Water at the base of plants to reduce evaporation."
        )

    if rain_expected and not raining_now:
        notes.append(
            f"🌧 RAIN EXPECTED: {int(rain_prob_24h*100)}% chance in the next 24 hours. "
            "Delay any fertiliser or pesticide application until after the rain."
        )

    if not spray_ok and not raining_now:
        notes.append(
            f"💨 HIGH WINDS ({wind_kmh} km/h): Spraying pesticides/herbicides not recommended. "
            "Wait for winds below 15 km/h for effective and safe application."
        )

    if spray_ok and not high_humidity:
        notes.append(
            "✅ SPRAY CONDITIONS OK: Low wind and no rain — good window for pesticide/fungicide application."
        )

    if not notes:
        notes.append("🌿 Conditions look normal. Continue your regular farming schedule.")

    return {
        "irrigation_needed"       : irrigation_needed,
        "frost_risk"              : frost_risk,
        "heat_stress_risk"        : heat_stress,
        "rain_expected_24h"       : rain_expected,
        "spray_conditions_ok"     : spray_ok,
        "high_humidity_disease_risk": high_humidity,
        "advisory_notes"          : notes,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK DATA (no API key / development mode)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_response(location_input: str) -> dict:
    """
    Return realistic demo weather data for Indian agricultural conditions
    (Kharif season, Maharashtra plateau baseline).
    Used when no API key is configured.
    """
    # Determine a plausible display name
    is_pin  = _is_valid_pin(str(location_input))
    city    = "Nashik" if is_pin else str(location_input).title()
    pin_out = str(location_input) if is_pin else "422001"

    current = {
        "temp_c"          : 31.4,
        "feels_like_c"    : 34.2,
        "temp_min_c"      : 28.1,
        "temp_max_c"      : 33.8,
        "humidity_pct"    : 72,
        "pressure_hpa"    : 1009,
        "wind_speed_kmh"  : 13.0,
        "wind_direction"  : "NW",
        "visibility_km"   : 10.0,
        "cloud_cover_pct" : 40,
        "condition"       : "Partly cloudy",
        "condition_code"  : 802,
        "icon_code"       : "02d",
        "uv_index"        : None,
        "rainfall_1h_mm"  : 0.0,
        "rainfall_3h_mm"  : 0.0,
    }

    from datetime import date, timedelta as td
    today = date.today()
    forecast = [
        {
            "date"             : str(today + td(days=i)),
            "day_name"         : (today + td(days=i)).strftime("%A"),
            "temp_min_c"       : round(27.0 + i * 0.3, 1),
            "temp_max_c"       : round(33.0 + i * 0.2, 1),
            "humidity_pct"     : min(72 + i * 3, 95),
            "rainfall_mm"      : round(i * 1.5, 2),
            "rain_probability" : round(min(0.1 + i * 0.12, 0.9), 2),
            "condition"        : ["Partly cloudy", "Mostly cloudy",
                                   "Light rain", "Moderate rain", "Overcast"][i],
            "icon_code"        : ["02d", "03d", "10d", "10d", "04d"][i],
        }
        for i in range(5)
    ]

    advisory = _build_farming_advisory(current, forecast)

    return {
        "location": {
            "city"    : city,
            "state"   : "Maharashtra",
            "country" : "IN",
            "pin_code": pin_out,
            "lat"     : 19.9975,
            "lon"     : 73.7898,
        },
        "current"          : current,
        "forecast"         : forecast,
        "farming_advisory" : advisory,
        "fetched_at"       : _ist_now_str(),
        "source"           : "mock",
        "error"            : None,
        "_mock"            : True,   # Flag for UI to show "Demo Data" badge
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FETCH FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_all(lat: float, lon: float, api_key: str,
               session: requests.Session, pin_code: str = None) -> dict:
    """
    Internal: given coordinates, call both current + forecast endpoints
    and assemble the full response dict.
    """
    params_base = {
        "lat"   : lat,
        "lon"   : lon,
        "appid" : api_key,
        "units" : "metric",    # Always return Celsius / m/s
        "lang"  : "en",
    }

    # ── Current weather ───────────────────────────────────────────────────────
    try:
        resp_current = session.get(BASE_URL_CURRENT, params=params_base,
                                   timeout=REQUEST_TIMEOUT_SEC)
        resp_current.raise_for_status()
        raw_current = resp_current.json()
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"Current weather fetch failed (HTTP {e.response.status_code}): "
                         f"{e.response.text}")
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out fetching current weather.")
    except requests.exceptions.ConnectionError:
        raise ValueError("No internet connection. Check your network.")

    # ── 5-day forecast ────────────────────────────────────────────────────────
    try:
        resp_forecast = session.get(BASE_URL_FORECAST, params={**params_base, "cnt": 40},
                                    timeout=REQUEST_TIMEOUT_SEC)
        resp_forecast.raise_for_status()
        raw_forecast = resp_forecast.json()
        forecast_list = _parse_forecast(raw_forecast)
    except Exception:
        # Forecast failure is non-fatal — return current data with empty forecast
        forecast_list = []

    # ── Parse and assemble ────────────────────────────────────────────────────
    parsed = _parse_current(raw_current, pin_code=pin_code)
    advisory = _build_farming_advisory(parsed["current"], forecast_list)

    return {
        "location"         : parsed["location"],
        "current"          : parsed["current"],
        "forecast"         : forecast_list,
        "farming_advisory" : advisory,
        "fetched_at"       : _ist_now_str(),
        "source"           : "openweathermap",
        "error"            : None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_weather_by_pin(pin_code: Union[str, int], api_key: str) -> dict:
    """
    Fetch weather for a location identified by an Indian PIN code.

    This first geocodes the PIN → lat/lon, then fetches weather by coordinates.
    Falls back to mock data if no valid API key is provided.

    Args:
        pin_code : 6-digit Indian PIN code, e.g. "411001" or 411001
        api_key  : OpenWeatherMap API key

    Returns:
        Full weather dict (see module docstring for schema).

    Example:
        >>> data = get_weather_by_pin("411001", api_key="your_key")
        >>> print(data["location"]["city"])
        'Pune'
        >>> print(data["current"]["temp_c"], "°C")
        31.4 °C
    """
    pin_str = str(pin_code).strip()

    # ── Validate PIN format ───────────────────────────────────────────────────
    if not _is_valid_pin(pin_str):
        return _error_response(
            f"'{pin_str}' is not a valid Indian PIN code. "
            f"PIN codes are exactly 6 digits and start with 1–9.",
            pin_code=pin_str,
        )

    # ── No API key → mock ─────────────────────────────────────────────────────
    if not api_key or api_key.strip() == _PLACEHOLDER_KEY:
        print(f"[WARNING] No API key provided. Returning mock weather for PIN {pin_str}.")
        return _mock_response(pin_str)

    session = _make_session()

    try:
        # Step 1: PIN → coordinates
        coords = _resolve_pin_to_coords(pin_str, api_key, session)

        # Step 2: coordinates → weather
        result = _fetch_all(
            lat=coords["lat"], lon=coords["lon"],
            api_key=api_key, session=session, pin_code=pin_str,
        )

        # Enrich location with geocoded city name if OWM returned "Unknown"
        if result["location"]["city"] in ("", "Unknown") and coords.get("city"):
            result["location"]["city"] = coords["city"]

        return result

    except ValueError as e:
        return _error_response(str(e), pin_code=pin_str)
    except Exception as e:
        return _error_response(f"Unexpected error: {e}", pin_code=pin_str)


def get_weather_by_city(city: str, api_key: str, state: str = "") -> dict:
    """
    Fetch weather for an Indian city by name.

    Automatically appends ',IN' to bias results to India.

    Args:
        city    : City name, e.g. "Nashik", "Hyderabad", "Bengaluru"
        api_key : OpenWeatherMap API key
        state   : Optional state name to disambiguate (e.g. "Maharashtra")

    Returns:
        Full weather dict (see module docstring for schema).

    Example:
        >>> data = get_weather_by_city("Nashik", api_key="your_key")
        >>> print(data["current"]["humidity_pct"], "%")
        72 %
    """
    city = city.strip()
    if not city:
        return _error_response("City name cannot be empty.")

    # ── No API key → mock ─────────────────────────────────────────────────────
    if not api_key or api_key.strip() == _PLACEHOLDER_KEY:
        print(f"[WARNING] No API key provided. Returning mock weather for '{city}'.")
        return _mock_response(city)

    session = _make_session()

    # Build query: append state if given, always append country code
    query = f"{city},{state},IN" if state else f"{city},IN"

    params = {
        "q"     : query,
        "appid" : api_key,
        "units" : "metric",
        "lang"  : "en",
    }

    try:
        # Direct city lookup for current weather
        resp = session.get(BASE_URL_CURRENT, params=params, timeout=REQUEST_TIMEOUT_SEC)

        if resp.status_code == 404:
            return _error_response(
                f"City '{city}' not found. Try the nearest large city or use a PIN code.",
                city=city,
            )
        if resp.status_code == 401:
            return _error_response(
                "Invalid API key. Check your OPENWEATHER_API_KEY "
                "(new keys take ~10 minutes to activate).",
                city=city,
            )

        resp.raise_for_status()
        raw = resp.json()
        coord = raw.get("coord", {})

        # Now fetch full data (including forecast) using coordinates
        result = _fetch_all(
            lat=coord["lat"], lon=coord["lon"],
            api_key=api_key, session=session,
        )
        return result

    except ValueError as e:
        return _error_response(str(e), city=city)
    except requests.exceptions.Timeout:
        return _error_response("Request timed out. Please try again.", city=city)
    except requests.exceptions.ConnectionError:
        return _error_response("No internet connection. Check your network.", city=city)
    except Exception as e:
        return _error_response(f"Unexpected error: {e}", city=city)


def get_weather(location: Union[str, int], api_key: str) -> dict:
    """
    Unified entry point — auto-detects whether `location` is a PIN code or city name.

    This is the function to call from app.py and advisor.py.

    Args:
        location : PIN code (e.g. "411001") OR city name (e.g. "Pune")
        api_key  : OpenWeatherMap API key

    Returns:
        Full weather dict (see module docstring for schema).

    Examples:
        >>> get_weather("411001", api_key)   # PIN code
        >>> get_weather("Nashik", api_key)   # City name
        >>> get_weather(560001, api_key)     # INT pin code
    """
    if _is_valid_pin(str(location)):
        return get_weather_by_pin(str(location), api_key)
    else:
        return get_weather_by_city(str(location), api_key)


def get_forecast(location: Union[str, int], api_key: str) -> list:
    """
    Convenience wrapper — returns only the 5-day forecast list.

    Args:
        location : PIN code or city name
        api_key  : OpenWeatherMap API key

    Returns:
        List of daily forecast dicts (empty list on error).
    """
    result = get_weather(location, api_key)
    return result.get("forecast", [])


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_human(data: dict, show_forecast: bool = True):
    """Pretty-print weather data to the terminal."""
    W = 62
    print("\n" + "="*W)
    print("  🌦  Agri-Edge — Hyperlocal Weather Report")
    print("="*W)

    if data.get("error"):
        print(f"\n  ❌  ERROR: {data['error']}\n")
        return

    loc = data.get("location", {})
    cur = data.get("current", {})
    adv = data.get("farming_advisory", {})

    mock_flag = " [DEMO DATA — add API key for real weather]" if data.get("_mock") else ""
    pin_str = f"(PIN: {loc['pin_code']})" if loc.get('pin_code') else ""
    print(f"\n  📍  {loc.get('city', '?')}, {loc.get('state', loc.get('country', ''))} {pin_str}")
    print(f"  🕐  {data.get('fetched_at', '')}{mock_flag}")
    print(f"\n  {'─'*56}")
    print(f"  🌡  Temperature    : {cur.get('temp_c')}°C  "
          f"(feels like {cur.get('feels_like_c')}°C)")
    print(f"      Range         : {cur.get('temp_min_c')}°C – {cur.get('temp_max_c')}°C")
    print(f"  💧  Humidity       : {cur.get('humidity_pct')}%")
    print(f"  🌬  Wind           : {cur.get('wind_speed_kmh')} km/h {cur.get('wind_direction', '')}")
    print(f"  ☁️   Cloud Cover    : {cur.get('cloud_cover_pct')}%")
    print(f"  👁   Visibility    : {cur.get('visibility_km')} km")
    print(f"  🌧  Rainfall (1h)  : {cur.get('rainfall_1h_mm')} mm")
    print(f"  📡  Condition      : {cur.get('condition')}")
    print(f"  🔴  Pressure       : {cur.get('pressure_hpa')} hPa")

    if show_forecast and data.get("forecast"):
        print(f"\n  {'─'*56}")
        print(f"  📅  5-Day Forecast:")
        print(f"  {'─'*56}")
        print(f"  {'Day':<12} {'Min':>6} {'Max':>6} {'Hum':>5} {'Rain':>8}  Condition")
        print(f"  {'─'*56}")
        for d in data["forecast"]:
            rain_prob = f"({int(d['rain_probability']*100)}%)"
            print(
                f"  {d['day_name'][:3]:<4} {d['date'][5:]:<8}"
                f" {d['temp_min_c']:>5}°"
                f" {d['temp_max_c']:>5}°"
                f" {d['humidity_pct']:>4}%"
                f" {d['rainfall_mm']:>5}mm {rain_prob:<5}"
                f"  {d['condition']}"
            )

    if adv:
        print(f"\n  {'─'*56}")
        print(f"  🌾  Farming Advisory:")
        for note in adv.get("advisory_notes", []):
            print(f"      {note}")
        print(f"\n  Flags:")
        flags = [
            ("Irrigation needed",       adv.get("irrigation_needed")),
            ("Frost risk",              adv.get("frost_risk")),
            ("Heat stress risk",        adv.get("heat_stress_risk")),
            ("Rain expected (24h)",     adv.get("rain_expected_24h")),
            ("Spraying conditions OK",  adv.get("spray_conditions_ok")),
            ("Fungal disease risk",     adv.get("high_humidity_disease_risk")),
        ]
        for label, val in flags:
            icon = "✅" if val else "❌"
            print(f"      {icon}  {label}")

    print("\n" + "="*W + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Agri-Edge: Fetch hyperlocal weather by Indian PIN code or city.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/weather_fetcher.py --pin 411001
  python utils/weather_fetcher.py --city Nashik
  python utils/weather_fetcher.py --pin 560001 --json
  python utils/weather_fetcher.py --pin 110001 --no-forecast
  python utils/weather_fetcher.py --city Hyderabad --state Telangana
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin",  "-p", type=str, help="Indian PIN code (6 digits)")
    group.add_argument("--city", "-c", type=str, help="City name, e.g. 'Nashik'")

    parser.add_argument("--state",       "-s", type=str, default="",
                        help="State name to disambiguate city (optional)")
    parser.add_argument("--key",         "-k", type=str, default=None,
                        help="OpenWeatherMap API key (overrides env var)")
    parser.add_argument("--json",        "-j", action="store_true",
                        help="Output raw JSON")
    parser.add_argument("--no-forecast", "-nf", action="store_true",
                        help="Skip 5-day forecast (faster, uses 1 API call)")

    args = parser.parse_args()

    # Resolve API key: CLI arg > env var > placeholder
    api_key = (
        args.key
        or os.environ.get("OPENWEATHER_API_KEY", "")
        or _PLACEHOLDER_KEY
    )

    # Fetch
    if args.pin:
        data = get_weather_by_pin(args.pin, api_key)
    else:
        data = get_weather_by_city(args.city, api_key, state=args.state)

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        _print_human(data, show_forecast=not args.no_forecast)

    if data.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
