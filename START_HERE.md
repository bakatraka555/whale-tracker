# 🚀 START HERE - ONE-CLICK GUIDE

**Project:** Whale Tracker - 48h Validation Sprint  
**Goal:** Prove ML model works BEFORE building full app  
**Your repo:** https://github.com/bakatraka555/whale-tracker

---

## ⚡ QUICK START (Choose one):

### OPTION A: Demo Mode (NO API KEYS!) ⭐ EASIEST

**Test everything with synthetic data:**

```bash
# 1. Setup (first time only)
cd C:\Users\bakat\Desktop\tapthemap\whale-tracker
python -m venv venv
venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Generate demo data (30 seconds)
python scripts/generate_demo_data.py

# 3. Analyze patterns (30 seconds)
python scripts/analyze_patterns.py
python scripts/rank_whales.py

# 4. Prepare for ML (30 seconds)
python scripts/prepare_ml_data.py

# 5. Push to GitHub for Colab training
git add data/
git commit -m "Add demo training data"
git push
```

**Then go to:** [Colab Training](#colab-training-30-60min)

---

### OPTION B: Real Data (Requires Etherscan API)

**For production-ready results:**

```bash
# 1. Get API key
# Go to: https://etherscan.io/apis
# Sign up (free), create API key

# 2. Create .env file
# Add: ETHERSCAN_API_KEY=your_key_here

# 3. Collect real whale data (1-2 hours)
python scripts/collect_whales.py

# 4. Continue with Option A steps 3-5
```

---

## 🔥 COLAB TRAINING (30-60min)

**1. Open Colab:**  
👉 **https://colab.research.google.com/**

**2. Set GPU:**  
Runtime → Change runtime type → **T4 GPU** ✅

**3. Copy-paste these 6 cells:**

---

**CELL 1: Check GPU**
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

**CELL 2: Clone Repo**
```python
!git clone https://github.com/bakatraka555/whale-tracker.git
%cd whale-tracker
```

---

**CELL 3: Install**
```python
!pip install -q torch pandas numpy matplotlib seaborn scikit-learn
```

---

**CELL 4: Check Data**
```python
import os
files = ['data/X_train.npy', 'data/X_test.npy', 'data/y_train.npy', 'data/y_test.npy']
for f in files:
    print(f"{'✅' if os.path.exists(f) else '❌'} {f}")
```

**⚠️ IF MISSING:** Run Option A locally first!

---

**CELL 5: Train Model** ☕ (30-60min)
```python
!python scripts/train.py
```

---

**CELL 6: Download Results**
```python
from google.colab import files
files.download('models/whale_lstm_best.pth')
files.download('models/training_curves.png')
```

---

## 🎯 AFTER TRAINING (Local)

```bash
# Move downloaded files to:
# whale_lstm_best.pth → models/
# training_curves.png → models/

# Test accuracy
python scripts/test_accuracy.py

# Backtest trading
python scripts/backtest_trading.py
```

---

## ✅ SUCCESS CRITERIA:

**IF BOTH:**
- ✅ Accuracy ≥ 65%
- ✅ Backtest ROI ≥ 50%

**THEN: BUILD THE APP!** 🚀

**ELSE:**
- Need more data
- Or try pattern-based approach (no ML)

---

## 📞 CURRENT STATUS:

✅ GitHub repo created  
✅ All code pushed  
✅ Demo data generator ready  
✅ Colab instructions ready  

**NEXT:** Run Option A (demo mode) to test pipeline!

---

## 🐛 HELP:

**"Missing data files in Colab"**
→ Run `python scripts/generate_demo_data.py` locally first

**"No GPU in Colab"**
→ Runtime → Change runtime type → T4 GPU

**"Training fails"**
→ Check Cell 4 output (all files present?)

---

**READY TO START?**

**EASIEST PATH:**
1. Run `python scripts/generate_demo_data.py` (30 sec)
2. Run `python scripts/prepare_ml_data.py` (30 sec)
3. Push to GitHub
4. Open Colab (link above)
5. Copy-paste 6 cells
6. Wait 30-60min ☕
7. Check results!

**GO!** 🚀

