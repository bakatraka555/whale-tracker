"""
Trading Strategy Backtest Script
Simulates trading based on whale predictions to calculate ROI
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('.')
from train import WhaleLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def backtest_strategy():
    print("=" * 60)
    print("💰 TRADING STRATEGY BACKTEST - DAY 2")
    print("=" * 60)
    print()
    
    # Load whale data
    try:
        df = pd.read_csv('data/whale_data.csv')
        print(f"✅ Loaded {len(df)} whale transactions\n")
    except FileNotFoundError:
        print("❌ whale_data.csv not found!")
        return
    
    # Load model and scaler
    try:
        model = WhaleLSTM(input_size=6, hidden_size=64, num_layers=2).to(device)
        model.load_state_dict(torch.load('models/whale_lstm_best.pth'))
        model.eval()
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("✅ Model and scaler loaded\n")
    except FileNotFoundError:
        print("❌ Model not found! Run training first.")
        return
    
    # Trading parameters
    STARTING_CAPITAL = 10000  # $10,000
    POSITION_SIZE = 0.05  # 5% of portfolio per trade
    STOP_LOSS = 0.97  # -3%
    TAKE_PROFIT = 1.08  # +8%
    MAX_HOLD_DAYS = 7
    
    print("📋 Trading Parameters:")
    print(f"   Starting Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"   Position Size: {POSITION_SIZE*100:.0f}% per trade")
    print(f"   Stop Loss: -{(1-STOP_LOSS)*100:.0f}%")
    print(f"   Take Profit: +{(TAKE_PROFIT-1)*100:.0f}%")
    print(f"   Max Hold: {MAX_HOLD_DAYS} days")
    print()
    
    # Prepare features
    feature_columns = ['from_exchange', 'to_exchange', 'value_log', 'fear_greed', 'hour', 'day_of_week']
    df['value_log'] = np.log10(df['value_usd'])
    df['timestamp_dt'] = pd.to_datetime(df['date'])
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    
    # Simulate trading
    portfolio = STARTING_CAPITAL
    trades = []
    
    print("🏃 Running backtest simulation...")
    print()
    
    for idx, row in df.iterrows():
        # Skip if not enough future data
        if idx + MAX_HOLD_DAYS >= len(df):
            continue
        
        # Prepare features for prediction
        features = row[feature_columns].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(1).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        # Only trade if high confidence (>0.6 = 60%+ confidence)
        if prediction > 0.6:  # BULLISH signal
            # Entry
            entry_price = row['eth_price_at_tx']
            position_value = portfolio * POSITION_SIZE
            
            # Check next 7 days for exit
            future_prices = df.iloc[idx+1:idx+MAX_HOLD_DAYS+1]['eth_price_24h'].values
            
            if len(future_prices) == 0:
                continue
            
            # Determine exit
            exit_price = entry_price
            exit_reason = 'HOLD_7D'
            
            for day_idx, price in enumerate(future_prices):
                # Stop-loss hit
                if price <= entry_price * STOP_LOSS:
                    exit_price = entry_price * STOP_LOSS
                    exit_reason = 'STOP_LOSS'
                    break
                
                # Take-profit hit
                if price >= entry_price * TAKE_PROFIT:
                    exit_price = entry_price * TAKE_PROFIT
                    exit_reason = 'TAKE_PROFIT'
                    break
                
                # Last day
                if day_idx == len(future_prices) - 1:
                    exit_price = price
                    exit_reason = 'HOLD_7D'
            
            # Calculate P&L
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = position_value * (pnl_pct / 100)
            portfolio += pnl_usd
            
            trades.append({
                'date': row['date'],
                'prediction_conf': prediction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'portfolio': portfolio,
                'exit_reason': exit_reason
            })
    
    # Results
    if not trades:
        print("❌ No trades executed (insufficient high-confidence signals)")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate statistics
    total_return = ((portfolio - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    winners = trades_df[trades_df['pnl_usd'] > 0]
    losers = trades_df[trades_df['pnl_usd'] < 0]
    win_rate = len(winners) / len(trades_df) * 100
    
    avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
    
    max_drawdown = ((trades_df['portfolio'].min() - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    
    sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n💰 PERFORMANCE:")
    print(f"   Starting Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"   Ending Portfolio: ${portfolio:,.0f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Absolute Profit: ${portfolio - STARTING_CAPITAL:+,.0f}")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"   Losers: {len(losers)} ({100-win_rate:.1f}%)")
    
    print(f"\n💵 WIN/LOSS METRICS:")
    print(f"   Avg Win: {avg_win:+.2f}%")
    print(f"   Avg Loss: {avg_loss:+.2f}%")
    print(f"   Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "   Win/Loss Ratio: N/A")
    
    print(f"\n📉 RISK METRICS:")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    
    # Plot equity curve
    print(f"\n📈 Generating equity curve...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df['portfolio'], linewidth=2)
    plt.axhline(y=STARTING_CAPITAL, color='r', linestyle='--', label='Starting Capital')
    plt.fill_between(range(len(trades_df)), STARTING_CAPITAL, trades_df['portfolio'], 
                     where=trades_df['portfolio'] >= STARTING_CAPITAL, alpha=0.3, color='green')
    plt.fill_between(range(len(trades_df)), STARTING_CAPITAL, trades_df['portfolio'], 
                     where=trades_df['portfolio'] < STARTING_CAPITAL, alpha=0.3, color='red')
    plt.title(f'Portfolio Value Over Time (Total Return: {total_return:+.2f}%)')
    plt.xlabel('Trade Number')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/equity_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to models/equity_curve.png")
    
    # Save trades log
    trades_df.to_csv('data/backtest_trades.csv', index=False)
    print(f"\n💾 Trade log saved to data/backtest_trades.csv")
    
    # Final decision
    print("\n" + "=" * 60)
    print("🎯 BACKTEST VERDICT")
    print("=" * 60)
    
    if total_return >= 50 and win_rate >= 60:
        print(f"\n✅✅ EXCELLENT! Profitable strategy!")
        print(f"   Return: {total_return:+.2f}%")
        print(f"   Win rate: {win_rate:.1f}%")
        print("\n🚀 RECOMMENDATION: BUILD THE APP!")
    elif total_return >= 30 and win_rate >= 55:
        print(f"\n✅ GOOD: Profitable, but marginal")
        print(f"   Return: {total_return:+.2f}%")
        print(f"   Win rate: {win_rate:.1f}%")
        print("\n🤔 RECOMMENDATION: Proceed with caution")
    else:
        print(f"\n❌ BELOW TARGET:")
        print(f"   Return: {total_return:+.2f}%")
        print(f"   Win rate: {win_rate:.1f}%")
        print("\n🛑 RECOMMENDATION: Improve model or pivot")
    
    print("\n" + "=" * 60)
    print("✅ BACKTEST COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/generate_report.py")

if __name__ == "__main__":
    backtest_strategy()

