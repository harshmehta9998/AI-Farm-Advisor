# 🌾 Agri-Edge — AI-Powered Agricultural Advisory System

> Built for Indian farmers. Powered by Computer Vision + LLMs.

Agri-Edge helps farmers make smarter decisions by combining:
- 🔬 **Crop Disease Detection** via MobileNetV2 transfer learning
- 🌦 **Hyperlocal Weather** from OpenWeather API
- 📊 **Mandi Market Prices** (mocked, easily swappable with real API)
- 🤖 **AI Recommendations** via Groq LLM (Llama 3)

---

## 📁 Project Structure

```
agri_edge/
│
├── app.py                          # 🚀 Streamlit entry point (run this)
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # API keys, model paths, constants
│
├── models/
│   ├── __init__.py
│   ├── disease_detector.py         # MobileNetV2 inference logic
│   ├── train_model.py              # Transfer learning training script
│   └── saved_model/
│       ├── disease_model.h5        # Trained model (generate with train_model.py)
│       ├── class_labels.json       # Class index → disease name mapping
│       └── README.md
│
├── utils/
│   ├── __init__.py
│   ├── weather_fetcher.py          # OpenWeather API integration
│   ├── market_prices.py            # Mandi price data (mocked)
│   └── advisor.py                  # Groq LLM advisory generator
│
├── assets/
│   ├── icons/                      # App icons and logos
│   └── sample_images/              # Test leaf images for demo
│
├── pages/                          # (Reserved) Multi-page Streamlit expansion
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme (green agricultural theme)
│
├── .env.example                    # API key template — copy to .env
├── .gitignore
├── requirements.txt
└── README.md                       # ← You are here
```

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone / Download the project

```bash
git clone https://github.com/your-username/agri-edge.git
cd agri_edge
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon Mac?** Use instead:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> pip install -r requirements.txt
> ```

### 4. Configure API keys

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your keys:
# OPENWEATHER_API_KEY = get free at https://openweathermap.org/api
# GROQ_API_KEY        = get free at https://console.groq.com
```

> ✅ **No API keys? No problem!** The app runs in **Demo Mode** with mock data.
> All features are functional — just with simulated responses.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🤖 Getting API Keys (Free)

| Service      | Where to Get                          | Free Tier                  |
|------------- |-------------------------------------- |--------------------------- |
| OpenWeather  | https://openweathermap.org/api        | 1,000 calls/day            |
| Groq         | https://console.groq.com             | ~14,400 tokens/min free    |

---

## 🧠 Training Your Own Disease Detection Model

The app ships with **mock predictions** by default.
To use a real model:

### Step 1: Download the PlantVillage Dataset
```
https://www.kaggle.com/datasets/emmarex/plantdisease
```
Extract to: `data/plantvillage/`

### Step 2: Train the model
```bash
python models/train_model.py
```
This uses MobileNetV2 (ImageNet weights) with 2-phase transfer learning:
- Phase 1: Train classification head (5 epochs, LR=1e-3)
- Phase 2: Fine-tune top 30 MobileNetV2 layers (5 epochs, LR=1e-5)

Output: `models/saved_model/disease_model.h5`

### Step 3: Restart the app
```bash
streamlit run app.py
```
The detector automatically loads the `.h5` model if it exists.

---

## 🔌 Extending Agri-Edge

### Add a real Mandi price API
Replace `utils/market_prices.py` → `get_market_prices()` with a call to:
- **Agmarknet**: https://agmarknet.gov.in/
- **data.gov.in Open API**: https://data.gov.in/

### Switch LLM provider
In `utils/advisor.py`, change:
```python
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
```
To:
```python
# OpenAI
GROQ_BASE_URL = "https://api.openai.com/v1/chat/completions"
# Or use Gemini via their OpenAI-compatible endpoint
```

### Add more pages (Streamlit multi-page)
Drop a `.py` file in the `pages/` folder:
```
pages/
├── 1_market_overview.py
├── 2_crop_calendar.py
└── 3_soil_guide.py
```
Streamlit auto-discovers these as sidebar navigation pages.

---

## 🛠 Tech Stack

| Component        | Technology                        |
|----------------- |---------------------------------- |
| Frontend         | Streamlit                         |
| Disease Model    | TensorFlow / Keras + MobileNetV2  |
| LLM Advisory     | Groq API (Llama 3)                |
| Weather Data     | OpenWeather Current Weather API   |
| Market Prices    | Mocked (Agmarknet-compatible)     |
| Configuration    | python-dotenv                     |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ for Indian farmers 🇮🇳*
