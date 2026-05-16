"""
utils/market_prices.py
----------------------
Mocked Mandi (agricultural market) price data for common Indian crops.

Data structure mirrors the Government of India's Agmarknet API format.
Real API: https://agmarknet.gov.in/

To swap in real data later, replace get_market_prices() with an API call
to Agmarknet or data.gov.in and keep the same return schema.

Prices are in ₹ per quintal (100 kg).
"""

from datetime import date

# ── Mock Mandi Price Database ─────────────────────────────────────────────────
# Format: crop → {modal_price, min_price, max_price, market, state}
MANDI_PRICES = {
    "Tomato": {
        "modal_price": 1200,
        "min_price"  : 800,
        "max_price"  : 1800,
        "market"     : "Azadpur Mandi, Delhi",
        "state"      : "Delhi",
        "trend"      : "rising",   # rising | falling | stable
        "unit"       : "quintal",
    },
    "Potato": {
        "modal_price": 900,
        "min_price"  : 700,
        "max_price"  : 1100,
        "market"     : "Agra Mandi, UP",
        "state"      : "Uttar Pradesh",
        "trend"      : "stable",
        "unit"       : "quintal",
    },
    "Rice": {
        "modal_price": 2200,
        "min_price"  : 1900,
        "max_price"  : 2500,
        "market"     : "Karnal Mandi, Haryana",
        "state"      : "Haryana",
        "trend"      : "stable",
        "unit"       : "quintal",
    },
    "Wheat": {
        "modal_price": 2150,
        "min_price"  : 2000,
        "max_price"  : 2300,
        "market"     : "Indore Mandi, MP",
        "state"      : "Madhya Pradesh",
        "trend"      : "rising",
        "unit"       : "quintal",
    },
    "Cotton": {
        "modal_price": 6500,
        "min_price"  : 6000,
        "max_price"  : 7200,
        "market"     : "Akola Mandi, Maharashtra",
        "state"      : "Maharashtra",
        "trend"      : "falling",
        "unit"       : "quintal",
    },
    "Maize": {
        "modal_price": 1850,
        "min_price"  : 1650,
        "max_price"  : 2100,
        "market"     : "Davangere Mandi, Karnataka",
        "state"      : "Karnataka",
        "trend"      : "rising",
        "unit"       : "quintal",
    },
}

# MSP (Minimum Support Price) for Kharif 2024 — Government guaranteed prices
MSP_2024 = {
    "Rice"   : 2300,
    "Maize"  : 2090,
    "Cotton" : 7121,   # Medium staple
    "Wheat"  : 2275,   # Rabi MSP
}


def get_market_prices(crop: str) -> dict:
    """
    Get mandi price data for a given crop.

    Args:
        crop: Crop name (must match keys in MANDI_PRICES)

    Returns:
        dict with modal_price, min_price, max_price, market, date, msp
    """
    crop = crop.strip().title()
    prices = MANDI_PRICES.get(crop, _default_prices(crop))

    return {
        **prices,
        "crop" : crop,
        "date" : date.today().strftime("%d %b %Y"),
        "msp"  : MSP_2024.get(crop, None),   # None if MSP not available
    }


def get_all_prices() -> dict:
    """Return all crop prices — useful for a market overview page."""
    return {
        crop: get_market_prices(crop)
        for crop in MANDI_PRICES
    }


def _default_prices(crop: str) -> dict:
    """Fallback prices for unsupported crops."""
    return {
        "modal_price" : 1500,
        "min_price"   : 1200,
        "max_price"   : 1800,
        "market"      : "Local Mandi",
        "state"       : "India",
        "trend"       : "stable",
        "unit"        : "quintal",
    }
