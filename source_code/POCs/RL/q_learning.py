from pandas import read_csv
import datetime
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import numpy
 

#df.head()
#df.tail()

def rsi(df):
    '''This function builds Relative Strength Indicator.
    Returns:
    -------------
    DataFrame of stock prices and RSI
    '''
    delta = df['detrended_price_adj'].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up2 = up.rolling(14).mean()
    roll_down2 = down.abs().rolling(14).mean()

    rsi_raw = roll_up2/roll_down2
    rsi = 100.0 - (100.0 / (1.0 + rsi_raw))

    df['RSI'] = rsi

    return df[['Date', 'Close', 'detrended_price_adj', 'RSI','Volume']]

def macd(df):

    df['ema_price_12'] = df['detrended_price_adj'].ewm(span = 12, adjust = True, ignore_na=True).mean()
    df['ema_price_26'] = df['detrended_price_adj'].ewm(span = 26, adjust = True, ignore_na=True).mean()
    
    df['macd'] = df['ema_price_12'] - df['ema_price_26']
    df['signal_line'] = df['macd'].ewm(span = 9, adjust = True, ignore_na=True).mean()
    
    df['macd_diff_signal'] = df['macd'] - df['signal_line']
    
    return df[['Date', 'Close', 'macd', 'signal_line','macd_diff_signal','Volume', 'RSI','detrended_price_adj']]



def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('POCs/RL/apple.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
series = series.reset_index()
# fit linear model

X = [i for i in range(0, len(series))]
X = numpy.reshape(X, (len(X), 1))
y = series.Close
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
pyplot.plot(y)
pyplot.plot(trend)
pyplot.show()
# detrend
detrended = [y[i]-trend[i] for i in range(0, len(series))]
# plot detrended
pyplot.plot(detrended)
pyplot.show()

detrended_df = pd.merge(series, pd.DataFrame(detrended), how='left', left_index=True, right_index=True)
detrended_df = detrended_df.rename(columns={0:'detrended_price'})
detrended_df['detrended_price_adj'] = detrended_df['detrended_price'] + abs(detrended_df['detrended_price'].min())

df = rsi(detrended_df)
df = macd(df)

# plot the RSI and detrended adjusted price
df.plot('Date', ['detrended_price_adj','RSI']) # correlation makes sense still
df.plot('Date', ['detrended_price_adj','Close']) # comparing the detrended adjusted price to the closing price


## clean up the code above to spit out clean data that the RL agent can work with
## define the states
## define the actions
## define the reward function
## define the q-matrix and how to calculate it
## define the update the method 


def state_action_pair():

    states = ['RSI<25', 'RSI<50','RSI<75']
    
    q_table = pd.DataFrame(states).rename(columns={0:'states'})
    
    q_table['B'],q_table['S'],q_table['H'] = 0,0,0

    return q_table



def get_reward(df, i, j):

    if df[df.index==i]['detrended_price_adj'].values[0] > df[df.index==j]['detrended_price_adj'].values[0]:
        return 1
    elif df[df.index==i]['detrended_price_adj'].values[0] > df[df.index==j]['detrended_price_adj'].values[0]:
        return 0
    else:
        return -1
            
#    
#for i in df.index[1:10]:
#    j = i-1
#    print(get_reward(df, i, j))
#        
    
def update_q_values():
    """
    A function that updates the q values using the bellman equation and replaces the q_0 from the initial q-table
    """
        
