import pandas as pd, numpy as np
import tiingo, datetime
import warnings

warnings.filterwarnings('ignore')

def connect_to_tiingo():
    '''This function establishes a connection to the Tiingo API to donwload historical stock data.'''
    API = 'cb927ca36374f6d4f3455280cb187e4a59fb68f3'
    config = {}
    config['session'] = True
    config['api_key'] = API
    client = tiingo.TiingoClient(config)
    return client

def get_stock_data(ticker, tiingo_conn, startDate=datetime.datetime.now().date() - datetime.timedelta(days=365), endDate=datetime.datetime.now().date(), fmt='json', frequency='daily'):
    '''This function downloads the specified stock data. The function downloads date, close price and volume on the day
    of. The start and end date of time range are preset at January 1, 2018 and November 2, 2018 respectively, with
    daily frequency. The data is returned in json format.
    '''
    stock_data = tiingo_conn.get_ticker_price(ticker, startDate, endDate, fmt, frequency)
    stock_data = pd.DataFrame(stock_data)
    stock_data['date'] = pd.to_datetime(stock_data['date'],format='%Y-%m-%dT%H:%M:%S.%fz')
    stock_data = stock_data.reset_index().set_index(stock_data['date'])
    stock_data['ticker'] = ticker
    return stock_data[['date', 'close', 'volume', 'ticker','volume']]

def macd(df):
    '''This function builds MACD and the 9 day Signal line.

    Parameters:
    -------------
    plot: set to "no" by default. If "yes", outputs a graph of MACD and Signal line

    Returns:
    -------------
    DataFrame of MACD and Signal Line
    '''

    df['ema_price_12'] = df['close'].ewm(span = 12, adjust = True, ignore_na=True).mean()
    df['ema_price_26'] = df['close'].ewm(span = 26, adjust = True, ignore_na=True).mean()
    df['macd'] = df['ema_price_12'] - df['ema_price_26']
    df['signal_line'] = df['macd'].ewm(span = 9, adjust = True, ignore_na=True).mean()
    df['macd_diff_signal'] = df['macd'] - df['signal_line']

    return df[['date', 'close', 'macd', 'signal_line','macd_diff_signal', 'ticker','volume']]

def rsi(df):
    '''This function builds Relative Strength Indicator.

    Returns:
    -------------
    DataFrame of stock prices and RSI
    '''
    delta = df['close'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up2 = up.rolling(14).mean()
    roll_down2 = down.abs().rolling(14).mean()

    rsi_raw = roll_up2/roll_down2
    rsi = 100.0 - (100.0 / (1.0 + rsi_raw))
    df['rsi'] = rsi
    return df[['date', 'close', 'macd', 'signal_line', 'macd_diff_signal','rsi', 'ticker','volume']]

def set_above_below_indicator(df):
    # find the day over day change in macd_diff_signal, first difference
    df['delta'] = df['macd_diff_signal'].diff()

    # find the second difference, to measure % increase/decrease in day over day delta
    #.....

    # find global min
    global_min = df['macd_diff_signal'].min()

    # above vs below the global min
    df['gl_min_ab_be'] = np.where(df['macd_diff_signal']<=global_min,'below','above')

    return df

def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='high',
                       low_col='low', close_col='close', vol_col='volume'):
    ac = pd.Series([])
    val_last = 0
    
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac.set_value(index, val)
    val_last = val

    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    data['ch_osc'] = ema_short - ema_long

    return data

def action_signal_assigner(df):
    #1.1a: macd_diff below 0, above global min, declined last day (BUY / PAY ATTENTION)
    #1.1b: macd_diff below 0, above global min, increased last day (SELL / HOLD)
    #1.2a: macd_diff below 0, below global min, declined last day (BUY / PAY ATTENTION)
    #1.2b: macd_diff below 0, below global min, increased last day (SELL / HOLD)
    df['action_signal'] = np.where((((df['gl_min_ab_be'] == 'above') & (df['delta'] < 0) & (df['rsi'] < 30) & (df['macd_diff_signal'] < 0)) | 
                                   ((df['gl_min_ab_be'] == 'below') & (df['delta'] < 0) & (df['rsi'] < 30)& (df['macd_diff_signal'] < 0))), 'Bullish',                                   
                                   np.where(((df['gl_min_ab_be'] == 'above') & (df['delta'] > 0) & (df['rsi'] < 30) & (df['macd_diff_signal'] < 0)) | 
                                   ((df['gl_min_ab_be'] == 'below') & (df['delta'] > 0) & (df['rsi'] < 30) & (df['macd_diff_signal'] < 0)), 
                                    'Bearish', 'No signal'))   
    return df

def main():
    connection = connect_to_tiingo()

    stacked = pd.DataFrame(columns=['date','close','rsi','macd_diff_signal','macd','signal_line', 'ticker','volume'])
    stocks = pd.read_excel('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/ticker_data.xlsx')
    for i in stocks[stocks['Industry']!='Precious Metals']['Ticker']:
        try:
            df = get_stock_data(i, connection)
            df = macd(df)
            df = rsi(df)
            df = df.iloc[:, :-1]
            df = set_above_below_indicator(df)
            df = action_signal_assigner(df)
            stacked = stacked.append(df)
            stacked['date'] = stacked.index
        except:
            print(str(i), ' encountered error, moving to next stock')
            pass

    return stacked.to_csv('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/daily_updated_prices.csv', index=False)

if __name__ == "__main__":
    main()