"""
Demo Data Generator
Creates synthetic whale transaction data for testing WITHOUT API keys
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_demo_data():
    print("=" * 60)
    print("🎮 DEMO DATA GENERATOR")
    print("=" * 60)
    print("\n⚡ Creating synthetic whale transactions...")
    print("   (No API keys needed!)\n")
    
    # Configuration
    NUM_TRANSACTIONS = 500
    START_DATE = datetime(2024, 1, 1)
    END_DATE = datetime(2025, 1, 26)
    
    # Known exchange addresses (realistic)
    exchanges = {
        '0x28c6c06298d514db089934071355e5743bf21d60': 'Binance',
        '0xbe0eb53f46cd790cd13851d5eff43d12404d33e8': 'Binance Cold',
        '0x71660c4005ba85c37ccec55d0c4493e66fe775d3': 'Coinbase',
        '0x503828976d22510aad0201ac7ec88293211d23da': 'Coinbase 2',
        '0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0': 'Kraken',
    }
    
    whale_data = []
    
    # Realistic price patterns
    base_price = 2500  # ETH price around $2500
    
    for i in range(NUM_TRANSACTIONS):
        # Random timestamp
        days_offset = random.randint(0, (END_DATE - START_DATE).days)
        timestamp = START_DATE + timedelta(days=days_offset)
        unix_timestamp = int(timestamp.timestamp())
        
        # Random addresses
        from_addr = random.choice(list(exchanges.keys()))
        
        # 70% outflow (from exchange), 30% inflow (to exchange)
        is_outflow = random.random() < 0.7
        
        if is_outflow:
            to_addr = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
            from_exchange = 1
            to_exchange = 0
            from_label = exchanges[from_addr]
            to_label = 'Unknown'
        else:
            to_addr = from_addr
            from_addr = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
            from_exchange = 0
            to_exchange = 1
            from_label = 'Unknown'
            to_label = exchanges[to_addr]
        
        # Whale transaction size ($10M - $100M)
        value_usd = random.uniform(10_000_000, 100_000_000)
        
        # Calculate ETH amount
        eth_price = base_price + random.gauss(0, 300)  # Price fluctuation
        value_eth = value_usd / eth_price
        
        # Fear & Greed (0-100)
        fear_greed = random.randint(15, 85)
        
        # Price change 24h later
        # Pattern: Outflow during fear = bullish (realistic)
        if is_outflow and fear_greed < 40:
            # Accumulation during fear = usually bullish
            price_change_24h = random.gauss(3, 2)  # avg +3%
        elif is_outflow and fear_greed > 60:
            # Accumulation during greed = neutral/bearish
            price_change_24h = random.gauss(-1, 3)  # avg -1%
        elif not is_outflow and fear_greed > 70:
            # Distribution during greed = bearish
            price_change_24h = random.gauss(-4, 2)  # avg -4%
        else:
            # Random
            price_change_24h = random.gauss(0, 3)
        
        eth_price_24h = eth_price * (1 + price_change_24h / 100)
        
        label = 'UP' if price_change_24h > 0 else 'DOWN'
        
        # Hour and day features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        whale_data.append({
            'timestamp': unix_timestamp,
            'date': timestamp.strftime('%Y-%m-%d %H:%M'),
            'tx_hash': '0x' + ''.join(random.choices('0123456789abcdef', k=64)),
            'from_address': from_addr,
            'to_address': to_addr,
            'from_label': from_label,
            'to_label': to_label,
            'from_exchange': from_exchange,
            'to_exchange': to_exchange,
            'value_eth': value_eth,
            'value_usd': value_usd,
            'eth_price_at_tx': eth_price,
            'eth_price_24h': eth_price_24h,
            'price_change_24h': price_change_24h,
            'fear_greed': fear_greed,
            'label': label,
            'hour': hour,
            'day_of_week': day_of_week
        })
    
    # Save to CSV
    df = pd.DataFrame(whale_data)
    df.to_csv('data/whale_data.csv', index=False)
    
    # Print statistics
    print(f"✅ Generated {len(df)} transactions\n")
    print("📊 Dataset Statistics:")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   UP moves: {len(df[df['label'] == 'UP'])} ({len(df[df['label'] == 'UP'])/len(df)*100:.1f}%)")
    print(f"   DOWN moves: {len(df[df['label'] == 'DOWN'])} ({len(df[df['label'] == 'DOWN'])/len(df)*100:.1f}%)")
    print(f"   Avg whale size: ${df['value_usd'].mean():,.0f}")
    print(f"   Largest whale: ${df['value_usd'].max():,.0f}")
    print(f"   Avg price change: {df['price_change_24h'].mean():+.2f}%")
    
    # Pattern preview
    print("\n📋 Pattern Preview:")
    outflow_fear = df[(df['from_exchange'] == 1) & (df['fear_greed'] < 40)]
    if len(outflow_fear) > 0:
        acc = (outflow_fear['label'] == 'UP').sum() / len(outflow_fear) * 100
        print(f"   Outflow + Fear: {acc:.1f}% bullish ({len(outflow_fear)} samples)")
    
    inflow_greed = df[(df['to_exchange'] == 1) & (df['fear_greed'] > 70)]
    if len(inflow_greed) > 0:
        acc = (inflow_greed['label'] == 'DOWN').sum() / len(inflow_greed) * 100
        print(f"   Inflow + Greed: {acc:.1f}% bearish ({len(inflow_greed)} samples)")
    
    print("\n💾 Saved to: data/whale_data.csv")
    print("\n" + "=" * 60)
    print("✅ DEMO DATA READY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. python scripts/analyze_patterns.py")
    print("2. python scripts/rank_whales.py")
    print("3. python scripts/prepare_ml_data.py")
    print("4. Use Colab for training (GPU)")

if __name__ == "__main__":
    generate_demo_data()

