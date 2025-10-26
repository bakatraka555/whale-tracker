"""
Whale Wallet Ranking Script
Identifies the most accurate whale wallets (Tier S/A/B classification)
"""

import pandas as pd
import numpy as np

def rank_whale_wallets():
    print("=" * 60)
    print("🏆 WHALE WALLET RANKING - DAY 1")
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
    
    # Track wallet performance
    whale_wallets = {}
    
    print("📊 Analyzing individual whale wallets...")
    print()
    
    for idx, tx in df.iterrows():
        # Focus on wallets that SEND (initiators of moves)
        wallet = tx['from_address']
        
        if wallet not in whale_wallets:
            whale_wallets[wallet] = {
                'wallet': wallet,
                'label': tx['from_label'],
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'avg_amount': 0,
                'accuracy': 0
            }
        
        # Track performance (only for outflows from exchanges = accumulation signals)
        if tx['from_exchange'] == 1:
            whale_wallets[wallet]['trades'] += 1
            
            if tx['label'] == 'UP':
                whale_wallets[wallet]['wins'] += 1
                whale_wallets[wallet]['total_pnl'] += tx['price_change_24h']
            else:
                whale_wallets[wallet]['losses'] += 1
                whale_wallets[wallet]['total_pnl'] += tx['price_change_24h']
            
            whale_wallets[wallet]['avg_amount'] += tx['value_usd']
    
    # Calculate statistics
    whale_stats = []
    for wallet, stats in whale_wallets.items():
        if stats['trades'] >= 3:  # Minimum 3 trades to be significant
            stats['accuracy'] = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            stats['avg_amount'] = stats['avg_amount'] / stats['trades'] if stats['trades'] > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            whale_stats.append(stats)
    
    # Sort by accuracy
    whale_stats_sorted = sorted(whale_stats, key=lambda x: x['accuracy'], reverse=True)
    
    # Classify into tiers
    tier_s = [w for w in whale_stats_sorted if w['trades'] >= 10 and w['accuracy'] >= 75]
    tier_a = [w for w in whale_stats_sorted if w['trades'] >= 7 and 65 <= w['accuracy'] < 75]
    tier_b = [w for w in whale_stats_sorted if w['trades'] >= 5 and 55 <= w['accuracy'] < 65]
    tier_c = [w for w in whale_stats_sorted if w['trades'] >= 3 and w['accuracy'] < 55]
    
    # Print results
    print("=" * 60)
    print("🏅 WHALE TIER CLASSIFICATION")
    print("=" * 60)
    
    print(f"\n⭐ TIER S (Elite - 75%+ accuracy): {len(tier_s)} whales")
    for i, whale in enumerate(tier_s[:5], 1):
        print(f"\n{i}. {whale['wallet'][:10]}... ({whale['label']})")
        print(f"   Accuracy: {whale['accuracy']:.1f}%")
        print(f"   Trades: {whale['trades']}")
        print(f"   Avg Amount: ${whale['avg_amount']:,.0f}")
        print(f"   Avg P&L: {whale['avg_pnl']:.2f}%")
    
    print(f"\n🥇 TIER A (Good - 65-75% accuracy): {len(tier_a)} whales")
    for i, whale in enumerate(tier_a[:3], 1):
        print(f"\n{i}. {whale['wallet'][:10]}... ({whale['label']})")
        print(f"   Accuracy: {whale['accuracy']:.1f}%")
        print(f"   Trades: {whale['trades']}")
    
    print(f"\n🥈 TIER B (Average - 55-65% accuracy): {len(tier_b)} whales")
    print(f"🥉 TIER C (Below average - <55% accuracy): {len(tier_c)} whales")
    
    # Save top whales
    top_whales_df = pd.DataFrame(whale_stats_sorted[:20])
    top_whales_df.to_csv('data/top_whales.csv', index=False)
    print(f"\n💾 Saved top 20 whales to data/top_whales.csv")
    
    # Calculate aggregate statistics
    print("\n" + "=" * 60)
    print("📊 AGGREGATE STATISTICS")
    print("=" * 60)
    
    if tier_s:
        tier_s_df = pd.DataFrame(tier_s)
        print(f"\n⭐ TIER S Average:")
        print(f"   Accuracy: {tier_s_df['accuracy'].mean():.1f}%")
        print(f"   Avg P&L: {tier_s_df['avg_pnl'].mean():.2f}%")
    
    if tier_a:
        tier_a_df = pd.DataFrame(tier_a)
        print(f"\n🥇 TIER A Average:")
        print(f"   Accuracy: {tier_a_df['accuracy'].mean():.1f}%")
        print(f"   Avg P&L: {tier_a_df['avg_pnl'].mean():.2f}%")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if tier_s:
        print(f"\n✅ Found {len(tier_s)} TIER S whales (75%+ accuracy)")
        print("   → Premium feature: 'Follow Elite Whales Only'")
        print(f"   → Expected accuracy: {tier_s_df['accuracy'].mean():.1f}%")
    else:
        print("\n⚠️ No TIER S whales found (need more data or lower threshold)")
    
    if tier_a:
        print(f"\n✅ Found {len(tier_a)} TIER A whales (65-75% accuracy)")
        print("   → Good for PRO tier alerts")
    
    total_trackable = len(tier_s) + len(tier_a) + len(tier_b)
    print(f"\n📊 Total trackable whales: {total_trackable}")
    print(f"   Combined accuracy: {pd.DataFrame(whale_stats_sorted[:total_trackable])['accuracy'].mean():.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ WHALE RANKING COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/prepare_ml_data.py")

if __name__ == "__main__":
    rank_whale_wallets()

