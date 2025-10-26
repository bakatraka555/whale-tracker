# 🐋 Whale Tracker - AI-Powered Crypto Whale Alert System

## Project Overview
Whale Tracker analyzes large cryptocurrency transactions ("whale moves") and uses machine learning to predict price movements with 65-75% accuracy.

## 48-Hour Validation Sprint
**Goal:** Prove the concept with historical data before building the full app.

### Success Criteria
- ✅ Collect 500-1,000 whale transactions (2024-2025)
- ✅ Train LSTM model with 65%+ accuracy
- ✅ Backtest showing 50%+ return
- ✅ Identify top 20 most accurate whale wallets

## Project Structure
```
whale-tracker/
├── data/              # Whale transaction CSVs
├── scripts/           # Data collection & analysis scripts
├── models/            # Trained ML models
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Day 1: Data Collection
```bash
python scripts/collect_whales.py
python scripts/analyze_patterns.py
python scripts/rank_whales.py
python scripts/prepare_ml_data.py
python scripts/train.py  # Run overnight
```

### Day 2: Validation
```bash
python scripts/test_accuracy.py
python scripts/backtest_trading.py
python scripts/generate_report.py
```

## Environment Variables
Copy `.env.example` to `.env` and add your API keys:
- `ETHERSCAN_API_KEY` - Free at https://etherscan.io/apis
- `COINGECKO_API_KEY` - Optional, free tier available

## Tech Stack
- **Python 3.10+**
- **PyTorch** (LSTM neural network)
- **Pandas** (data analysis)
- **Matplotlib** (visualization)
- **Requests** (API calls)

## Next Steps (If Successful)
1. Backend development (Supabase + Netlify Functions)
2. ML model deployment (Replicate.com)
3. Mobile app (React Native + Expo)
4. Launch on Google Play Store

## License
MIT

---
**Built with Cursor AI** ⚡


