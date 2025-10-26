"""
Pattern Analysis Script
Analyzes whale transaction patterns to find high-accuracy signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_patterns():
    print("=" * 60)
    print("📊 WHALE PATTERN ANALYSIS - DAY 1")
    print("=" * 60)
    print()
    
    # Load data
    try:
        df = pd.read_csv('data/whale_data.csv')
        print(f"✅ Loaded {len(df)} whale transactions\n")
    except FileNotFoundError:
        print("❌ whale_data.csv not found!")
        print("Run: python scripts/collect_whales.py first")
        return
    
    # 1. Exchange Flow Patterns
    print("=" * 60)
    print("🔄 EXCHANGE FLOW PATTERNS")
    print("=" * 60)
    
    # From Exchange → Unknown (Accumulation)
    from_ex = df[(df['from_exchange'] == 1) & (df['to_exchange'] == 0)]
    if len(from_ex) > 0:
        bullish_pct = (from_ex['label'] == 'UP').sum() / len(from_ex) * 100
        print(f"\n📤 From Exchange → Unknown:")
        print(f"   Bullish: {bullish_pct:.1f}% ({len(from_ex)} transactions)")
        print(f"   Avg price change: {from_ex['price_change_24h'].mean():.2f}%")
    
    # Unknown → To Exchange (Distribution)
    to_ex = df[(df['from_exchange'] == 0) & (df['to_exchange'] == 1)]
    if len(to_ex) > 0:
        bearish_pct = (to_ex['label'] == 'DOWN').sum() / len(to_ex) * 100
        print(f"\n📥 Unknown → To Exchange:")
        print(f"   Bearish: {bearish_pct:.1f}% ({len(to_ex)} transactions)")
        print(f"   Avg price change: {to_ex['price_change_24h'].mean():.2f}%")
    
    # 2. Whale Size Patterns
    print("\n" + "=" * 60)
    print("💰 WHALE SIZE PATTERNS")
    print("=" * 60)
    
    for threshold in [10_000_000, 25_000_000, 50_000_000, 100_000_000]:
        large = df[df['value_usd'] > threshold]
        if len(large) >= 5:
            bullish = (large['label'] == 'UP').sum() / len(large) * 100
            print(f"\n💎 >${threshold/1_000_000:.0f}M transactions:")
            print(f"   Bullish: {bullish:.1f}% ({len(large)} transactions)")
            print(f"   Avg price change: {large['price_change_24h'].mean():.2f}%")
    
    # 3. Fear & Greed Correlation
    print("\n" + "=" * 60)
    print("😨😄 FEAR & GREED PATTERNS")
    print("=" * 60)
    
    fear_levels = [
        (0, 25, "Extreme Fear"),
        (25, 45, "Fear"),
        (45, 55, "Neutral"),
        (55, 75, "Greed"),
        (75, 100, "Extreme Greed")
    ]
    
    for low, high, label_name in fear_levels:
        subset = df[(df['fear_greed'] >= low) & (df['fear_greed'] < high)]
        if len(subset) >= 5:
            bullish = (subset['label'] == 'UP').sum() / len(subset) * 100
            print(f"\n{label_name} ({low}-{high}):")
            print(f"   Bullish: {bullish:.1f}% ({len(subset)} transactions)")
            print(f"   Avg price change: {subset['price_change_24h'].mean():.2f}%")
    
    # 4. COMBO PATTERNS (High Accuracy Signals)
    print("\n" + "=" * 60)
    print("🎯 COMBO PATTERNS (High Accuracy)")
    print("=" * 60)
    
    patterns = []
    
    # Pattern 1: Exchange outflow + Extreme Fear
    p1 = df[(df['from_exchange'] == 1) & (df['fear_greed'] < 25)]
    if len(p1) >= 5:
        acc = (p1['label'] == 'UP').sum() / len(p1) * 100
        patterns.append(('Exchange Outflow + Extreme Fear', acc, len(p1), p1['price_change_24h'].mean()))
        print(f"\n🔥 Exchange Outflow + Extreme Fear:")
        print(f"   Accuracy: {acc:.1f}% ({len(p1)} transactions)")
        print(f"   Avg gain: {p1['price_change_24h'].mean():.2f}%")
    
    # Pattern 2: Mega whale (>$50M) + Fear
    p2 = df[(df['value_usd'] > 50_000_000) & (df['fear_greed'] >= 25) & (df['fear_greed'] < 45)]
    if len(p2) >= 5:
        acc = (p2['label'] == 'UP').sum() / len(p2) * 100
        patterns.append(('Mega Whale + Fear', acc, len(p2), p2['price_change_24h'].mean()))
        print(f"\n💎 Mega Whale (>$50M) + Fear:")
        print(f"   Accuracy: {acc:.1f}% ({len(p2)} transactions)")
        print(f"   Avg gain: {p2['price_change_24h'].mean():.2f}%")
    
    # Pattern 3: Exchange inflow + Extreme Greed (Bearish!)
    p3 = df[(df['to_exchange'] == 1) & (df['fear_greed'] > 75)]
    if len(p3) >= 5:
        acc = (p3['label'] == 'DOWN').sum() / len(p3) * 100
        patterns.append(('Exchange Inflow + Extreme Greed', acc, len(p3), p3['price_change_24h'].mean()))
        print(f"\n⚠️ Exchange Inflow + Extreme Greed:")
        print(f"   Bearish accuracy: {acc:.1f}% ({len(p3)} transactions)")
        print(f"   Avg loss: {p3['price_change_24h'].mean():.2f}%")
    
    # 5. Save patterns to file
    patterns_df = pd.DataFrame(patterns, columns=['Pattern', 'Accuracy', 'Count', 'Avg_Change'])
    patterns_df = patterns_df.sort_values('Accuracy', ascending=False)
    patterns_df.to_csv('data/patterns_summary.csv', index=False)
    
    # 6. Visualizations
    print("\n" + "=" * 60)
    print("📈 Creating visualizations...")
    print("=" * 60)
    
    # Distribution plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('UP vs DOWN Distribution')
    plt.ylabel('Count')
    plt.xlabel('Label')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='price_change_24h', by='label', figsize=(6, 6))
    plt.title('Price Change by Label')
    plt.suptitle('')
    plt.ylabel('Price Change (%)')
    
    plt.tight_layout()
    plt.savefig('data/pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to data/pattern_analysis.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"\nTop 3 patterns by accuracy:")
    for idx, row in patterns_df.head(3).iterrows():
        print(f"{idx+1}. {row['Pattern']}: {row['Accuracy']:.1f}% ({row['Count']} samples)")
    
    print("\n" + "=" * 60)
    print("✅ PATTERN ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/rank_whales.py")

if __name__ == "__main__":
    analyze_patterns()

