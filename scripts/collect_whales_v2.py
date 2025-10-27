"""
Advanced Whale Data Collection - V2
Uses multiple Etherscan endpoints to find real whale moves:
- ERC-20 token transfers (USDT, USDC, WETH)
- Internal transactions
- Normal ETH transfers
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
WHALE_THRESHOLD_USD = 1_000_000  # $1M+

# Major stablecoins and wrapped ETH (where whales actually move money!)
MAJOR_TOKENS = {
    '0xdac17f958d2ee523a2206206994597c13d831ec7': {'name': 'USDT', 'decimals': 6, 'price_usd': 1.0},
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': {'name': 'USDC', 'decimals': 6, 'price_usd': 1.0},
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2': {'name': 'WETH', 'decimals': 18, 'price_usd': 2500},  # Approx ETH price
}

# Known exchanges
EXCHANGE_ADDRESSES = {
    '0x28c6c06298d514db089934071355e5743bf21d60': 'Binance',
    '0xbe0eb53f46cd790cd13851d5eff43d12404d33e8': 'Binance Cold',
    '0x71660c4005ba85c37ccec55d0c4493e66fe775d3': 'Coinbase',
    '0x503828976d22510aad0201ac7ec88293211d23da': 'Coinbase 2',
    '0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0': 'Kraken',
}

def classify_address(address):
    """Check if address is known exchange"""
    address_lower = address.lower()
    for exchange_addr, name in EXCHANGE_ADDRESSES.items():
        if exchange_addr.lower() == address_lower:
            return True, name
    return False, 'Unknown'

def get_latest_erc20_transfers(token_address, token_info, limit=500):
    """Get recent large ERC-20 token transfers"""
    print(f"  Scanning {token_info['name']} transfers...")
    
    url = "https://api.etherscan.io/api"
    params = {
        'module': 'logs',
        'action': 'getLogs',
        'address': token_address,
        'topic0': '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef',  # Transfer event
        'page': 1,
        'offset': limit,
        'apikey': ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        time.sleep(0.25)  # Rate limiting
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == '1' and 'result' in data:
                transfers = []
                
                for log in data['result']:
                    try:
                        # Decode transfer data
                        topics = log.get('topics', [])
                        if len(topics) < 3:
                            continue
                        
                        from_addr = '0x' + topics[1][-40:]
                        to_addr = '0x' + topics[2][-40:]
                        
                        # Value is in data field
                        value_hex = log.get('data', '0x0')
                        value_raw = int(value_hex, 16)
                        value_tokens = value_raw / (10 ** token_info['decimals'])
                        value_usd = value_tokens * token_info['price_usd']
                        
                        # Only large transfers
                        if value_usd < WHALE_THRESHOLD_USD:
                            continue
                        
                        # Classify addresses
                        from_is_exchange, from_label = classify_address(from_addr)
                        to_is_exchange, to_label = classify_address(to_addr)
                        
                        # Skip exchange-to-exchange (internal moves)
                        if from_is_exchange and to_is_exchange:
                            continue
                        
                        timestamp = int(log.get('timeStamp', '0'), 16)
                        
                        transfers.append({
                            'timestamp': timestamp,
                            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M'),
                            'tx_hash': log.get('transactionHash', ''),
                            'from_address': from_addr,
                            'to_address': to_addr,
                            'from_label': from_label,
                            'to_label': to_label,
                            'from_exchange': 1 if from_is_exchange else 0,
                            'to_exchange': 1 if to_is_exchange else 0,
                            'value_usd': value_usd,
                            'token': token_info['name'],
                        })
                        
                    except Exception as e:
                        continue
                
                print(f"    Found {len(transfers)} large {token_info['name']} transfers")
                return transfers
        
        return []
    
    except Exception as e:
        print(f"    Error: {e}")
        return []

def simulate_price_movement(timestamp, from_exchange, to_exchange, fear_greed):
    """
    Simulate realistic price movement based on whale action
    (In production, you'd fetch real historical price data)
    """
    import random
    random.seed(timestamp)
    
    # Realistic patterns based on research
    if from_exchange == 1 and to_exchange == 0:
        # Exchange outflow (accumulation)
        if fear_greed < 35:
            # Accumulation during fear = very bullish
            price_change = random.gauss(2.5, 1.5)  # avg +2.5%
        else:
            # Accumulation during normal/greed = neutral
            price_change = random.gauss(0.5, 2.0)
    
    elif from_exchange == 0 and to_exchange == 1:
        # Exchange inflow (distribution)
        if fear_greed > 65:
            # Distribution during greed = very bearish
            price_change = random.gauss(-2.5, 1.5)  # avg -2.5%
        else:
            # Distribution during normal/fear = neutral
            price_change = random.gauss(-0.5, 2.0)
    
    else:
        # Unknown to unknown
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
    except Exception as e:
        print(f"Error fetching Fear & Greed: {e}")
        return {}

def main():
    print("=" * 60)
    print("ADVANCED WHALE DATA COLLECTION - V2")
    print("=" * 60)
    print(f"\nThreshold: ${WHALE_THRESHOLD_USD:,}")
    print(f"Scanning ERC-20 tokens: {', '.join([t['name'] for t in MAJOR_TOKENS.values()])}\n")
    
    all_transfers = []
    
    # Collect from all major tokens
    for token_addr, token_info in MAJOR_TOKENS.items():
        transfers = get_latest_erc20_transfers(token_addr, token_info)
        all_transfers.extend(transfers)
        time.sleep(0.5)  # Rate limiting
    
    if not all_transfers:
        print("\nNo whale transactions found!")
        print("Possible reasons:")
        print("  - API key issue")
        print("  - Rate limit hit")
        print("  - No large transfers in recent blocks")
        return
    
    print(f"\nTotal whale transfers found: {len(all_transfers)}")
    
    # Add Fear & Greed index
    print("\nFetching Fear & Greed index...")
    fear_greed_data = get_fear_greed_index()
    
    # Process data
    for transfer in all_transfers:
        # Add Fear & Greed
        timestamp = transfer['timestamp']
        closest_fg = min(fear_greed_data.keys(), 
                        key=lambda x: abs(x - timestamp),
                        default=None)
        
        if closest_fg:
            transfer['fear_greed'] = fear_greed_data[closest_fg]
        else:
            transfer['fear_greed'] = 50
        
        # Simulate price movement (in production: real data)
        price_change = simulate_price_movement(
            timestamp,
            transfer['from_exchange'],
            transfer['to_exchange'],
            transfer['fear_greed']
        )
        
        transfer['price_change_24h'] = price_change
        transfer['label'] = 'UP' if price_change > 0 else 'DOWN'
        
        # Add time features
        dt = datetime.fromtimestamp(timestamp)
        transfer['hour'] = dt.hour
        transfer['day_of_week'] = dt.weekday()
    
    # Save to CSV
    df = pd.DataFrame(all_transfers)
    df.to_csv('data/whale_data.csv', index=False)
    
    print("\nDataset summary:")
    print(f"  Total transactions: {len(df)}")
    print(f"  UP moves: {len(df[df['label'] == 'UP'])} ({len(df[df['label'] == 'UP'])/len(df)*100:.1f}%)")
    print(f"  DOWN moves: {len(df[df['label'] == 'DOWN'])} ({len(df[df['label'] == 'DOWN'])/len(df)*100:.1f}%)")
    print(f"  Avg whale size: ${df['value_usd'].mean():,.0f}")
    print(f"  Largest whale: ${df['value_usd'].max():,.0f}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Show patterns
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
    print("DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/prepare_ml_data.py")

if __name__ == "__main__":
    main()

