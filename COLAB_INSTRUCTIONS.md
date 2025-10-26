# 🚀 Google Colab Training Instructions

## STEP 1: Push to GitHub (5min)

```bash
cd C:\Users\bakat\Desktop\tapthemap\whale-tracker

# Add all files
git add .

# Commit
git commit -m "Whale Tracker initial" --author="YourName <your@email.com>"

# Create GitHub repo:
# Go to: https://github.com/new
# Name: whale-tracker
# Public or Private

# After creating repo, GitHub gives you commands:
git remote add origin https://github.com/YOUR_USERNAME/whale-tracker.git
git branch -M main
git push -u origin main
```

---

## STEP 2: Run Data Collection LOCALLY (2-4h)

**Training data is too large for GitHub, but we need it!**

```bash
# Locally (on your laptop):
python scripts/collect_whales.py       # 1-2h
python scripts/prepare_ml_data.py      # 5min

# Upload ONLY processed data to GitHub:
git add data/*.npy models/scaler.pkl
git commit -m "Add training data"
git push
```

**Files to upload:**
- `data/X_train.npy` (~few KB)
- `data/X_test.npy` (~few KB)
- `data/y_train.npy` (~few KB)
- `data/y_test.npy` (~few KB)
- `models/scaler.pkl` (~few KB)

**Skip:** `data/whale_data.csv` (može biti veliki, nije potreban za training)

---

## STEP 3: Open Google Colab

**Go to:** https://colab.research.google.com/

**Create new notebook** → Copy-paste cells below!

---

## 📋 COLAB NOTEBOOK (Copy-paste ovo):

### CELL 1: Check GPU

```python
import torch
print(f"🖥️ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ NO GPU! Go to: Runtime → Change runtime type → GPU")
```

**⚠️ IF NO GPU:** Runtime → Change runtime type → T4 GPU

---

### CELL 2: Clone Your GitHub Repo

```python
# CHANGE THIS to your GitHub repo!
GITHUB_REPO = "https://github.com/YOUR_USERNAME/whale-tracker.git"

!git clone {GITHUB_REPO}
%cd whale-tracker
!ls -la
```

---

### CELL 3: Install Dependencies

```python
!pip install -q torch pandas numpy matplotlib seaborn scikit-learn python-dotenv tqdm
print("✅ Dependencies installed!")
```

---

### CELL 4: Check Data Files

```python
import os

required_files = [
    'data/X_train.npy',
    'data/X_test.npy',
    'data/y_train.npy',
    'data/y_test.npy',
    'models/scaler.pkl'
]

print("📂 Checking data files...")
missing = []
for f in required_files:
    if os.path.exists(f):
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} MISSING!")
        missing.append(f)

if missing:
    print("\n⚠️ Missing files! Run locally first:")
    print("   python scripts/collect_whales.py")
    print("   python scripts/prepare_ml_data.py")
    print("   git add data/ && git commit -m 'Add data' && git push")
    print("\nThen re-run this cell!")
else:
    print("\n✅ All data files present! Ready to train!")
```

---

### CELL 5: Train Model (30-60min) ☕

```python
!python scripts/train.py
```

**⏱️ This will take 30-60min. Go grab coffee!**

---

### CELL 6: Check Results

```python
from IPython.display import Image, display

print("📊 Training Curves:")
display(Image('models/training_curves.png'))

# Check model
import os
if os.path.exists('models/whale_lstm_best.pth'):
    print("\n✅ Model trained and saved!")
    print(f"   Size: {os.path.getsize('models/whale_lstm_best.pth') / 1e6:.2f} MB")
else:
    print("\n❌ Model not found! Check training errors above.")
```

---

### CELL 7: Download Trained Model

```python
from google.colab import files

# Download to your laptop
files.download('models/whale_lstm_best.pth')
files.download('models/training_curves.png')

print("✅ Files downloaded! Check your Downloads folder.")
```

---

### CELL 8: Quick Accuracy Test (Optional)

```python
!python scripts/test_accuracy.py
```

---

## STEP 4: Back to Laptop (Local Testing)

**After downloading model from Colab:**

```bash
# Move downloaded files to project:
# whale_lstm_best.pth → models/
# training_curves.png → models/

# Test accuracy
python scripts/test_accuracy.py

# Backtest trading strategy
python scripts/backtest_trading.py
```

---

## 🎯 SUCCESS CRITERIA:

**Test Accuracy:**
- ✅ 65%+ = EXCELLENT! Proceed!
- ⚠️ 60-65% = Good, but improve
- ❌ <60% = Need more data

**Backtest Return:**
- ✅ 50%+ ROI = Profitable!
- ⚠️ 30-50% = Marginal
- ❌ <30% = Not viable

---

## 🐛 TROUBLESHOOTING:

### "fatal: could not read Username"
GitHub repo is private, Colab can't access. Solution:
- Make repo Public (temporarily)
- Or use GitHub token

### "Missing data files"
You forgot to push training data! Run locally:
```bash
python scripts/collect_whales.py
python scripts/prepare_ml_data.py
git add data/*.npy models/scaler.pkl
git push
```

### Colab disconnects after 12h
- Free tier = 12h max
- Just re-run from CELL 5 (it will resume if checkpoint exists)

---

## 💡 PRO TIP:

**Monitor training in real-time:**
- Colab shows progress bars
- Look for "Best Validation Accuracy" updates
- Should see 60%+ by epoch 30-50

**Early stopping:**
- If accuracy plateaus below 60% after 30 epochs
- Stop training (Kernel → Interrupt)
- More data needed!

---

**READY? START WITH STEP 1!** 🚀

