"""
Whale Data Collection Script
Collects large crypto transactions from Etherscan + price data from CoinGecko
Target: 500-1,000 whale transactions from 2024-2025
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')

# Configuration
WHALE_THRESHOLD_USD = 10_000_000  # $10M+
START_DATE = '2024-01-01'
END_DATE = '2025-01-26'

# Known exchange addresses (for classification)
EXCHANGE_ADDRESSES = {
    '0x28c6c06298d514db089934071355e5743bf21d60': 'Binance Hot Wallet',
    '0xbe0eb53f46cd790cd13851d5eff43d12404d33e8': 'Binance Cold Wallet',
    '0x71660c4005ba85c37ccec55d0c4493e66fe775d3': 'Coinbase Wallet 1',
    '0x503828976d22510aad0201ac7ec88293211d23da': 'Coinbase Wallet 2',
    '0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0': 'Kraken Wallet',
    '0x0548f59fee79f8832c299e01dca5c76f034f558e': 'Bitfinex Wallet',
    # Add more as discovered
}

def get_eth_price_at_timestamp(timestamp):
    """Get historical ETH price from CoinGecko"""
    try:
        # CoinGecko requires timestamp in seconds
        url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': timestamp - 300,  # 5 min before
            'to': timestamp + 300      # 5 min after
        }
        
        headers = {}
        if COINGECKO_API_KEY:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('prices') and len(data['prices']) > 0:
                return data['prices'][0][1]  # Price in USD
        
        # Fallback: approximate price if API fails
        return None
    
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def classify_address(address):
    """Check if address is a known exchange"""
    address_lower = address.lower()
    for exchange_addr, name in EXCHANGE_ADDRESSES.items():
        if exchange_addr.lower() == address_lower:
            return True, name
    return False, 'Unknown'

def get_large_eth_transactions():
    """
    Collect large ETH transactions from Etherscan
    Note: This is a simplified version. For production, you'd need to:
    1. Track specific whale wallets
    2. Monitor ERC-20 token transfers
    3. Use multiple data sources
    """
    print("🐋 Collecting whale transactions from Etherscan...")
    print(f"Threshold: ${WHALE_THRESHOLD_USD:,}")
    print(f"Date range: {START_DATE} to {END_DATE}\n")
    
    whale_transactions = []
    
    # For demo: Fetch transactions from known whale wallets
    # In production: Monitor entire blockchain or use Whale Alert API
    
    whale_wallets_to_check = [
        '0x28c6c06298d514db089934071355e5743bf21d60',  # Binance
        '0xbe0eb53f46cd790cd13851d5eff43d12404d33e8',  # Binance Cold
        '0x71660c4005ba85c37ccec55d0c4493e66fe775d3',  # Coinbase
    ]
    
    for wallet in tqdm(whale_wallets_to_check, desc="Scanning whale wallets"):
        try:
            # Get transactions for this wallet
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': wallet,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 100,  # Last 100 transactions
                'sort': 'desc',
                'apikey': ETHERSCAN_API_KEY
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == '1' and 'result' in data:
                    for tx in data['result']:
                        try:
                            # Convert Wei to ETH
                            value_eth = int(tx['value']) / 1e18
                            
                            # Skip if too small
                            if value_eth < 1:  # Less than 1 ETH
                                continue
                            
                            # Get ETH price at transaction time
                            timestamp = int(tx['timeStamp'])
                            eth_price = get_eth_price_at_timestamp(timestamp)
                            
                            if not eth_price:
                                continue  # Skip if no price data
                            
                            value_usd = value_eth * eth_price
                            
                            # Check if whale transaction
                            if value_usd >= WHALE_THRESHOLD_USD:
                                # Classify from/to
                                from_is_exchange, from_label = classify_address(tx['from'])
                                to_is_exchange, to_label = classify_address(tx['to'])
                                
                                # Get price 24h later
                                timestamp_24h = timestamp + 86400
                                eth_price_24h = get_eth_price_at_timestamp(timestamp_24h)
                                
                                if eth_price_24h:
                                    price_change_24h = ((eth_price_24h - eth_price) / eth_price) * 100
                                    label = 'UP' if price_change_24h > 0 else 'DOWN'
                                    
                                    whale_transactions.append({
                                        'timestamp': timestamp,
                                        'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M'),
                                        'tx_hash': tx['hash'],
                                        'from_address': tx['from'],
                                        'to_address': tx['to'],
                                        'from_label': from_label,
                                        'to_label': to_label,
                                        'from_exchange': 1 if from_is_exchange else 0,
                                        'to_exchange': 1 if to_is_exchange else 0,
                                        'value_eth': value_eth,
                                        'value_usd': value_usd,
                                        'eth_price_at_tx': eth_price,
                                        'eth_price_24h': eth_price_24h,
                                        'price_change_24h': price_change_24h,
                                        'label': label
                                    })
                                    
                                    print(f"  ✅ Whale: ${value_usd:,.0f} | {from_label} → {to_label} | {label}")
                        
                        except Exception as e:
                            continue
            
            # Rate limiting (free tier = 5 calls/sec)
            time.sleep(0.3)
        
        except Exception as e:
            print(f"Error scanning {wallet}: {e}")
            continue
    
    return whale_transactions

def get_fear_greed_index():
    """Get historical Fear & Greed index"""
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
    print("🐋 WHALE DATA COLLECTION - DAY 1")
    print("=" * 60)
    print()
    
    # Step 1: Collect whale transactions
    whale_data = get_large_eth_transactions()
    
    if not whale_data:
        print("\n❌ No whale transactions collected!")
        print("Check your ETHERSCAN_API_KEY in .env file")
        return
    
    print(f"\n✅ Collected {len(whale_data)} whale transactions!")
    
    # Step 2: Add Fear & Greed index
    print("\n📊 Fetching Fear & Greed index...")
    fear_greed_data = get_fear_greed_index()
    
    if fear_greed_data:
        for tx in whale_data:
            # Find closest Fear & Greed value
            timestamp = tx['timestamp']
            closest_fg = min(fear_greed_data.keys(), 
                           key=lambda x: abs(x - timestamp),
                           default=None)
            
            if closest_fg:
                tx['fear_greed'] = fear_greed_data[closest_fg]
            else:
                tx['fear_greed'] = 50  # Neutral default
        
        print(f"✅ Added Fear & Greed data!")
    
    # Step 3: Save to CSV
    df = pd.DataFrame(whale_data)
    output_file = 'data/whale_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to {output_file}")
    print(f"\nDataset summary:")
    print(f"  Total transactions: {len(df)}")
    print(f"  UP moves: {len(df[df['label'] == 'UP'])} ({len(df[df['label'] == 'UP'])/len(df)*100:.1f}%)")
    print(f"  DOWN moves: {len(df[df['label'] == 'DOWN'])} ({len(df[df['label'] == 'DOWN'])/len(df)*100:.1f}%)")
    print(f"  Avg whale size: ${df['value_usd'].mean():,.0f}")
    print(f"  Largest whale: ${df['value_usd'].max():,.0f}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\n" + "=" * 60)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print("\nNext step: python scripts/analyze_patterns.py")

if __name__ == "__main__":
    main()

