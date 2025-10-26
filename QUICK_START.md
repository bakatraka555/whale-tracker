# ⚡ QUICK START - 48H VALIDATION SPRINT

## 🎯 CILJ
Dokazati da Whale Alerts koncept radi PRIJE nego gradimo cijelu app!

**Success criteria:**
- ✅ LSTM model sa 65%+ accuracy
- ✅ Backtest showing 50%+ return
- ✅ Identificirano 10+ top whale wallets

---

## 📋 DAY 1 - SETUP & DATA COLLECTION (8-10h)

### STEP 1: Environment Setup (30min)

```bash
# 1. Open terminal in whale-tracker folder
cd C:\Users\bakat\Desktop\tapthemap\whale-tracker

# 2. Create Python virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Check GPU (important!)
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
# Should print: GPU Available: True
```

**IF GPU = False:**
- Check CUDA installation
- Update NVIDIA drivers
- Reinstall PyTorch with CUDA support

---

### STEP 2: Get API Keys (15min)

**Etherscan API (REQUIRED):**
1. Go to: https://etherscan.io/apis
2. Sign up (free)
3. Create API key
4. Copy key

**Create `.env` file:**
```bash
# In whale-tracker folder, create file: .env
ETHERSCAN_API_KEY=YOUR_KEY_HERE
```

**CoinGecko API (OPTIONAL):**
- Free tier: No key needed
- Pro tier ($129/mo): Better rate limits (skip for now)

---

### STEP 3: Collect Whale Data (2-4h)

```bash
python scripts/collect_whales.py
```

**What happens:**
- Scans Binance, Coinbase whale wallets
- Fetches large transactions (>$10M)
- Gets ETH prices (at transaction + 24h later)
- Adds Fear & Greed index
- Saves to `data/whale_data.csv`

**Expected output:**
```
✅ Collected 50-200 whale transactions!
Dataset summary:
  UP moves: 60-65%
  Avg whale size: $25M-50M
```

**⏱️ Time:** 1-2h (rate limits slows it down)

---

### STEP 4: Analyze Patterns (30min)

```bash
python scripts/analyze_patterns.py
```

**What it shows:**
- Exchange flow patterns (accumulation vs distribution)
- Whale size correlations (bigger = more accurate?)
- Fear & Greed impact
- **High-accuracy combo patterns** (80%+ signals!)

**Look for:**
- Patterns with 70%+ accuracy
- Minimum 10+ samples per pattern

---

### STEP 5: Rank Whale Wallets (30min)

```bash
python scripts/rank_whales.py
```

**What it finds:**
- Top 20 most accurate whale wallets
- Tier S/A/B classification
- Individual whale performance

**Look for:**
- TIER S whales (75%+ accuracy)
- At least 5-10 good whales

---

### STEP 6: Prepare ML Data (30min)

```bash
python scripts/prepare_ml_data.py
```

**What it does:**
- Feature engineering
- Train/test split (80/20)
- Standardization
- Saves processed data for training

---

### STEP 7: Train LSTM Model (3-6h, OVERNIGHT!)

```bash
python scripts/train.py
```

**What happens:**
- Trains LSTM neural network
- 100 epochs (with early stopping)
- Saves best model
- Generates training curves

**⚡ LET THIS RUN OVERNIGHT!**

**Expected:**
- Training time: 2-4h (GPU) or 6-10h (CPU)
- Best accuracy: 63-72% (target: 65%+)

---

## 📋 DAY 2 - TESTING & VALIDATION (6-8h)

### STEP 8: Check Training Results (10min)

**Morning check:**
```bash
# Did training finish?
ls models/whale_lstm_best.pth

# If yes, check accuracy:
python scripts/test_accuracy.py
```

**Target:** 65%+ accuracy ✅

---

### STEP 9: Backtest Trading Strategy (2h)

```bash
python scripts/backtest_trading.py
```

**What it simulates:**
- $10,000 starting capital
- Following whale signals
- 5% position size
- Stop-loss + take-profit

**Target:**
- Total return: 50%+ ✅
- Win rate: 60%+ ✅
- Sharpe ratio: >1.5 ✅

---

### STEP 10: Review Results (2h)

**Check files created:**
```
models/
  - training_curves.png (accuracy graph)
  - confusion_matrix.png (prediction quality)
  - equity_curve.png (portfolio growth)

data/
  - whale_data.csv (raw data)
  - patterns_summary.csv (best patterns)
  - top_whales.csv (best wallets)
  - backtest_trades.csv (trade log)
```

---

## 🎯 DECISION TIME

### IF BOTH:
- ✅ LSTM accuracy ≥ 65%
- ✅ Backtest return ≥ 50%

**DECISION: GO! 🚀**
```
→ Subscribe to Cursor Pro ($20/mo)
→ Start Week 3 development (backend)
→ Build the app!
```

### IF ONE FAILS:
- ⚠️ Accuracy 60-65% OR Return 30-50%

**DECISION: IMPROVE 🔄**
```
→ Collect more data (extend to 2023)
→ Add features (volume, exchange netflow)
→ Re-train
→ Re-test in 2-3 days
```

### IF BOTH FAIL:
- ❌ Accuracy <60% AND Return <30%

**DECISION: PIVOT 🔄**
```
→ Try different ML approach
→ Or pivot to different idea
→ You saved months of work!
```

---

## 🐛 TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"
```bash
# Reduce batch size or use CPU
# Edit train.py: device = torch.device('cpu')
```

### "Etherscan API: Rate limit exceeded"
```bash
# Add delays in collect_whales.py
# time.sleep(0.5)  # Slow down requests
```

### "Not enough whale data collected"
```bash
# Extend date range or lower threshold
# Edit collect_whales.py:
# START_DATE = '2023-01-01'  # Go back further
# WHALE_THRESHOLD_USD = 5_000_000  # Lower threshold
```

---

## 📞 HELP

Zapeo si? Pitaj u chatu:
- "Model ne trenira" → Check GPU setup
- "Accuracy niska" → Need more data or features
- "Script error" → Share error message

---

## 🎉 SUCCESS CHECKLIST

After 48h, you should have:
- [ ] 500-1,000 whale transactions collected
- [ ] Patterns identified (70%+ accuracy patterns found)
- [ ] Top 20 whale wallets ranked
- [ ] LSTM model trained (65%+ accuracy)
- [ ] Backtest showing profitable strategy (50%+ return)
- [ ] Decision made: GO/NO-GO

**If GO → Next: Backend development (Week 3-4)**

---

**LET'S VALIDATE THIS IDEA! 🐋💰🚀**

