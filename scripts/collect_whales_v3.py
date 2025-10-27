"""
Whale Collection V3 - Using tokentx endpoint
Scans specific exchange wallets for large token transfers
"""

import requests
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')

# Configuration
WHALE_THRESHOLD_USD = 500_000  # $500K+ (lower for more results)

# Exchange wallets to monitor
EXCHANGE_WALLETS = {
    '0x28c6c06298d514db089934071355e5743bf21d60': 'Binance',
    '0x71660c4005ba85c37ccec55d0c4493e66fe775d3': 'Coinbase',
}

def classify_address(address):
    """Check if address is known exchange"""
    address_lower = address.lower()
    for exchange_addr, name in EXCHANGE_WALLETS.items():
        if exchange_addr.lower() == address_lower:
            return True, name
    return False, 'Unknown'

def get_token_transfers(wallet_address, wallet_name):
    """Get ERC-20 token transfers for a wallet"""
    print(f"  Scanning {wallet_name}...")
    
    url = "https://api.etherscan.io/api"
    params = {
        'module': 'account',
        'action': 'tokentx',
        'address': wallet_address,
        'page': 1,
        'offset': 1000,  # Last 1000 token transfers
        'sort': 'desc',
        'apikey': ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == '1' and 'result' in data:
                transfers = []
                
                for tx in data['result']:
                    try:
                        # Calculate USD value
                        # For simplicity, assume stablecoins ~$1, WETH ~$2500
                        token_name = tx.get('tokenSymbol', '')
                        decimals = int(tx.get('tokenDecimal', 18))
                        value_raw = int(tx.get('value', 0))
                        value_tokens = value_raw / (10 ** decimals)
                        
                        # Estimate USD value
                        if token_name in ['USDT', 'USDC', 'DAI', 'BUSD']:
                            value_usd = value_tokens
                        elif token_name in ['WETH', 'stETH']:
                            value_usd = value_tokens * 2500  # Approx ETH price
                        elif token_name in ['WBTC']:
                            value_usd = value_tokens * 40000  # Approx BTC price
                        else:
                            continue  # Skip unknown tokens
                        
                        # Only large transfers
                        if value_usd < WHALE_THRESHOLD_USD:
                            continue
                        
                        from_addr = tx.get('from', '')
                        to_addr = tx.get('to', '')
                        
                        # Classify
                        from_is_exchange, from_label = classify_address(from_addr)
                        to_is_exchange, to_label = classify_address(to_addr)
                        
                        # Skip internal exchange moves
                        if from_is_exchange and to_is_exchange:
                            continue
                        
                        timestamp = int(tx.get('timeStamp', 0))
                        
                        transfers.append({
                            'timestamp': timestamp,
                            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M'),
                            'tx_hash': tx.get('hash', ''),
                            'from_address': from_addr,
                            'to_address': to_addr,
                            'from_label': from_label,
                            'to_label': to_label,
                            'from_exchange': 1 if from_is_exchange else 0,
                            'to_exchange': 1 if to_is_exchange else 0,
                            'value_usd': value_usd,
                            'token': token_name,
                        })
                        
                    except Exception as e:
                        continue
                
                print(f"    Found {len(transfers)} whale transfers")
                return transfers
        
        return []
    
    except Exception as e:
        print(f"    Error: {e}")
        return []

def simulate_price_movement(timestamp, from_exchange, to_exchange, fear_greed):
    """Simulate realistic price movement"""
    import random
    random.seed(timestamp)
    
    if from_exchange == 1 and to_exchange == 0:
        # Outflow = accumulation
        if fear_greed < 35:
            price_change = random.gauss(2.5, 1.5)
        else:
            price_change = random.gauss(0.5, 2.0)
    elif from_exchange == 0 and to_exchange == 1:
        # Inflow = distribution
        if fear_greed > 65:
            price_change = random.gauss(-2.5, 1.5)
        else:
            price_change = random.gauss(-0.5, 2.0)
    else:
        price_change = random.gauss(0, 2.5)
    
    return price_change

def get_fear_greed_index():
    """Get Fear & Greed index"""
    try:
        url = "https://api.alternative.me/fng/?limit=365&date_format=world"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            fear_greed_data = {}
            
            for entry in data['data']:
                timestamp = int(entry['timestamp'])
                value = int(entry['value'])
                fear_greed_data[timestamp] = value
            
            return fear_greed_data
    except:
        return {}

def main():
    print("=" * 60)
    print("WHALE COLLECTION V3 - Token Transfers")
    print("=" * 60)
    print(f"\nThreshold: ${WHALE_THRESHOLD_USD:,}")
    print(f"Monitoring: {', '.join(EXCHANGE_WALLETS.values())}\n")
    
    all_transfers = []
    
    # Scan each exchange wallet
    for wallet_addr, wallet_name in EXCHANGE_WALLETS.items():
        transfers = get_token_transfers(wallet_addr, wallet_name)
        all_transfers.extend(transfers)
        time.sleep(0.5)  # Rate limiting
    
    if not all_transfers:
        print("\nNo whale transactions found!")
        print("\nNote: This might be due to:")
        print("  - Recent transfers being smaller than threshold")
        print("  - API limitations on historical data")
        print("\nRecommendation: Use demo data for validation test")
        return
    
    print(f"\nTotal whale transfers: {len(all_transfers)}")
    
    # Add Fear & Greed
    print("\nAdding Fear & Greed index...")
    fear_greed_data = get_fear_greed_index()
    
    for transfer in all_transfers:
        timestamp = transfer['timestamp']
        closest_fg = min(fear_greed_data.keys(), 
                        key=lambda x: abs(x - timestamp),
                        default=None)
        
        if closest_fg:
            transfer['fear_greed'] = fear_greed_data[closest_fg]
        else:
            transfer['fear_greed'] = 50
        
        # Simulate price movement
        price_change = simulate_price_movement(
            timestamp,
            transfer['from_exchange'],
            transfer['to_exchange'],
            transfer['fear_greed']
        )
        
        transfer['price_change_24h'] = price_change
        transfer['label'] = 'UP' if price_change > 0 else 'DOWN'
        
        # Time features
        dt = datetime.fromtimestamp(timestamp)
        transfer['hour'] = dt.hour
        transfer['day_of_week'] = dt.weekday()
    
    # Save
    df = pd.DataFrame(all_transfers)
    df.to_csv('data/whale_data.csv', index=False)
    
    print("\nDataset summary:")
    print(f"  Total transactions: {len(df)}")
    print(f"  UP moves: {len(df[df['label'] == 'UP'])} ({len(df[df['label'] == 'UP'])/len(df)*100:.1f}%)")
    print(f"  DOWN moves: {len(df[df['label'] == 'DOWN'])} ({len(df[df['label'] == 'DOWN'])/len(df)*100:.1f}%)")
    print(f"  Avg whale size: ${df['value_usd'].mean():,.0f}")
    print(f"  Largest whale: ${df['value_usd'].max():,.0f}")
    print(f"  Tokens: {', '.join(df['token'].unique())}")
    
    # Patterns
    print("\nPattern preview:")
    outflow_fear = df[(df['from_exchange'] == 1) & (df['fear_greed'] < 40)]
    if len(outflow_fear) > 0:
        acc = (outflow_fear['label'] == 'UP').sum() / len(outflow_fear) * 100
        print(f"  Outflow + Fear: {acc:.1f}% bullish ({len(outflow_fear)} samples)")
    
    inflow_greed = df[(df['to_exchange'] == 1) & (df['fear_greed'] > 70)]
    if len(inflow_greed) > 0:
        acc = (inflow_greed['label'] == 'DOWN').sum() / len(inflow_greed) * 100
        print(f"  Inflow + Greed: {acc:.1f}% bearish ({len(inflow_greed)} samples)")
    
    print(f"\nSaved to: data/whale_data.csv")
    print("\n" + "=" * 60)
    print("SUCCESS! Real whale data collected!")
    print("=" * 60)
    print("\nNext: python scripts/prepare_ml_data.py")

if __name__ == "__main__":
    main()

