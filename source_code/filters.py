import pandas as pd

def volume(df):
    
    df_volume_mean = pd.DataFrame(df.groupby('ticker').agg('mean')['volume'].astype('int')).rename(columns={'volume':'daily mean volume'})
    
    merged = pd.merge(df, df_volume_mean, how='inner', left_on='ticker', right_index=True)
    
    return merged

def main():
    df = pd.read_csv('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/daily_updated_prices.csv').round({'close':2, 'rsi':2, 'macd_diff_signal':2})    
    new_df = volume(df)

    return new_df.to_csv('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/daily_updated_prices.csv', index = False)

if __name__ == '__main__':
    main()