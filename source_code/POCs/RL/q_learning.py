import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import numpy
import random
 

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
 
series = pd.read_csv('POCs/RL/apple.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
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

# limit the df to work with manageable dataset for debugging
df = df.iloc[15:265].reset_index()

"""DONE"""
## clean up the code above to spit out clean data that the RL agent can work with
## define the states
## define the actions
## define the reward function
## define a function to identify the state the agent is
## define a function that lets the agent choose an action
## define the q-matrix and how to calculate it
## define the update the method 

"""TO BE COMPLETED"""
## change reward function from -1, 0 and 1 to a different measure
## add more states based on other indicators
## add exploration vs. exploitation rate instead of randomly choosing at step 1 then basing off q-value

def state_action_pair():

    states = ['RSI<25', 'RSI<50','RSI<75', 'RSI<100']
    
    q_table = pd.DataFrame(states).rename(columns={0:'states'})
    
    q_table['B'],q_table['S'],q_table['H'] = 0,0,0

    return q_table

q_table = state_action_pair()
q_table = q_table.set_index('states')    

def get_current_state(df, i):
    rsi = df.iloc[i]['RSI']

    if rsi < 25:
        rsi_state = 'RSI<25'
    elif rsi < 50:
        rsi_state = 'RSI<50'
    elif rsi < 75:
        rsi_state = 'RSI<75'
    elif rsi < 100:
        rsi_state = 'RSI<100'
    else:
        print('rsi unknown')
    
    return rsi_state

def exploration_vs_exploitation():
    """Will return whether the agent exploits the q-table or explores new actions at this step"""
    
    return None

def choose_action(q_table, rsi_state):
    
    if q_table.loc[rsi_state]['B']==q_table.loc[rsi_state]['H']==q_table.loc[rsi_state]['S']:
        action_choice = random.choice(['B','H','S'])
        print('chosen randomly ', action_choice)

    else:
        action_choice = q_table.loc['RSI<25'].idxmax(axis=1)
        print('chosen based on q_value ', action_choice)
    return action_choice

# At EOD of the trading day, calculate the reward of the action taken at the beginnning of the day
def get_reward(df, i, j):

    if df[df.index==i]['detrended_price_adj'].values[0] > df[df.index==j]['detrended_price_adj'].values[0]:
        return 1
    elif df[df.index==i]['detrended_price_adj'].values[0] == df[df.index==j]['detrended_price_adj'].values[0]:
        return 0
    else:
        return -1

q_table = state_action_pair()
q_table = q_table.set_index('states')    

ALPHA = 0.8
GAMMA = 0.5

pd.options.display.float_format = '{:.4f}'.format

for i in range(1,249):

    j = i-1
    
    reward = get_reward(df, i, j)
    
    rsi_state = get_current_state(df, i)
    print('the state is ', rsi_state)
    
    action_choice = choose_action(q_table, rsi_state)
    
    old_q_value = q_table.loc[rsi_state,action_choice]
    print('old q_value is ', old_q_value)
    
    new_q_value = old_q_value + ALPHA*(reward + GAMMA*old_q_value)
    print('new q_value is ', new_q_value)
    
    q_table.loc[rsi_state, action_choice] = new_q_value
    print(q_table)
    
    print('--------------------------------------------')
    
    