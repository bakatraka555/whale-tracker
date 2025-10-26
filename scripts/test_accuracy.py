"""
Model Testing Script
Tests trained LSTM model accuracy and generates confusion matrix
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import model architecture
import sys
sys.path.append('.')
from train import WhaleLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    print("=" * 60)
    print("🧪 MODEL TESTING - DAY 2")
    print("=" * 60)
    print()
    
    # Load test data
    print("📂 Loading test data...")
    try:
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        print(f"✅ Test samples: {len(X_test)}\n")
    except FileNotFoundError:
        print("❌ Test data not found!")
        return
    
    # Prepare data
    X_test = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Load model
    print("🤖 Loading trained model...")
    try:
        model = WhaleLSTM(input_size=6, hidden_size=64, num_layers=2).to(device)
        model.load_state_dict(torch.load('models/whale_lstm_best.pth'))
        model.eval()
        print("✅ Model loaded!\n")
    except FileNotFoundError:
        print("❌ Model not found!")
        print("Run: python scripts/train.py first")
        return
    
    # Make predictions
    print("🔮 Making predictions...")
    with torch.no_grad():
        predictions_prob = model(X_test).squeeze()
        predictions = (predictions_prob > 0.5).float()
    
    # Calculate metrics
    accuracy = (predictions == y_test_tensor).float().mean().item() * 100
    
    # Move to CPU for sklearn
    y_test_np = y_test
    predictions_np = predictions.cpu().numpy()
    
    # Confusion matrix
    cm = confusion_matrix(y_test_np, predictions_np)
    
    # Classification report
    report = classification_report(
        y_test_np, predictions_np, 
        target_names=['DOWN', 'UP'],
        output_dict=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
    print(f"\n📈 UP Predictions:")
    print(f"   Precision: {report['UP']['precision']*100:.1f}%")
    print(f"   Recall: {report['UP']['recall']*100:.1f}%")
    print(f"   F1-Score: {report['UP']['f1-score']:.3f}")
    
    print(f"\n📉 DOWN Predictions:")
    print(f"   Precision: {report['DOWN']['precision']*100:.1f}%")
    print(f"   Recall: {report['DOWN']['recall']*100:.1f}%")
    print(f"   F1-Score: {report['DOWN']['f1-score']:.3f}")
    
    # Visualize confusion matrix
    print(f"\n📊 Generating confusion matrix...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted DOWN', 'Predicted UP'],
                yticklabels=['Actual DOWN', 'Actual UP'])
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to models/confusion_matrix.png")
    
    # Decision
    print("\n" + "=" * 60)
    print("🎯 VALIDATION DECISION")
    print("=" * 60)
    
    if accuracy >= 65:
        print(f"\n✅✅ EXCELLENT! {accuracy:.2f}% accuracy")
        print("\n🚀 PROCEED TO BACKTEST")
        print("   Run: python scripts/backtest_trading.py")
    elif accuracy >= 60:
        print(f"\n⚠️ GOOD: {accuracy:.2f}% accuracy (borderline)")
        print("\n🔄 PROCEED WITH CAUTION")
        print("   Run backtest, but consider improvements")
    else:
        print(f"\n❌ BELOW TARGET: {accuracy:.2f}% accuracy")
        print("\n🛑 CONSIDER:")
        print("   - Collect more data")
        print("   - Add features (exchange netflow, volume)")
        print("   - Try different model architecture")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TESTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()

