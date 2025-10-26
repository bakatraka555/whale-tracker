"""
LSTM Model Training Script
Trains neural network to predict whale transaction outcomes
Target: 65%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# LSTM Model Architecture
class WhaleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(WhaleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def train_model():
    print("=" * 60)
    print("🚀 LSTM MODEL TRAINING - DAY 1 (OVERNIGHT)")
    print("=" * 60)
    print()
    
    # Load prepared data
    print("📂 Loading prepared data...")
    try:
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"✅ Train: {len(X_train)} samples")
        print(f"✅ Test: {len(X_test)} samples")
        print()
    except FileNotFoundError:
        print("❌ Prepared data not found!")
        print("Run: python scripts/prepare_ml_data.py first")
        return
    
    # Reshape for LSTM (batch_size, sequence_length, features)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add sequence dimension
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Move to GPU
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    
    print(f"🤖 Initializing LSTM model...")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Layers: {num_layers}")
    print()
    
    model = WhaleLSTM(input_size, hidden_size, num_layers).to(device)
    
    # Training setup
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training parameters
    epochs = 100
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    
    print("=" * 60)
    print(f"🏋️ TRAINING START ({epochs} epochs)")
    print("=" * 60)
    print()
    print("Epoch | Train Loss | Val Accuracy | Best Acc | LR")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_predictions = (val_outputs.squeeze() > 0.5).float()
            val_accuracy = (val_predictions == y_test).float().mean().item() * 100
        
        # Track metrics
        train_losses.append(loss.item())
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"{epoch+1:3d}/{epochs} | {loss.item():10.4f} | {val_accuracy:11.2f}% | {best_accuracy:8.2f}% | {current_lr:.6f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/whale_lstm_best.pth')
            patience_counter = 0
            
            if (epoch + 1) % 5 == 0:
                print(f"     💾 Saved new best model! (Accuracy: {best_accuracy:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    print("\n" + "=" * 60)
    print("📊 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"📉 Final Training Loss: {train_losses[-1]:.4f}")
    print(f"⏱️ Total Epochs: {epoch + 1}")
    
    # Plot training curves
    print(f"\n📈 Generating training curves...")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.axhline(y=65, color='r', linestyle='--', label='Target 65%')
    plt.axhline(y=best_accuracy, color='g', linestyle='--', label=f'Best {best_accuracy:.1f}%')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to models/training_curves.png")
    
    # Decision
    print("\n" + "=" * 60)
    print("🎯 VALIDATION DECISION")
    print("=" * 60)
    
    if best_accuracy >= 65:
        print(f"\n✅ SUCCESS! Accuracy: {best_accuracy:.2f}% (>= 65% target)")
        print("\n🚀 RECOMMENDATION: Proceed with app development!")
        print("   - Model is viable for production")
        print("   - Expected user value: HIGH")
        print("   - Go-to-market: APPROVED")
    elif best_accuracy >= 60:
        print(f"\n⚠️ BORDERLINE: Accuracy: {best_accuracy:.2f}% (60-65%)")
        print("\n🔄 RECOMMENDATION: Improve model before proceeding")
        print("   - Collect more data (extend to 2023)")
        print("   - Add features (exchange netflow, volume)")
        print("   - Try different architectures")
    else:
        print(f"\n❌ BELOW TARGET: Accuracy: {best_accuracy:.2f}% (< 60%)")
        print("\n🛑 RECOMMENDATION: Pivot or improve significantly")
        print("   - Concept may not be viable with current approach")
        print("   - Consider alternative strategies")
    
    print("\n" + "=" * 60)
    print(f"💾 Model saved to: models/whale_lstm_best.pth")
    print("=" * 60)
    print("\nNext step (Day 2): python scripts/test_accuracy.py")

if __name__ == "__main__":
    prepare_ml_data()
    train_model()

