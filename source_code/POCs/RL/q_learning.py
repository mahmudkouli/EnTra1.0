import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import numpy as np
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
X = np.reshape(X, (len(X), 1))
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
df = df.iloc[15:1071].reset_index()

"""DONE"""
## clean up the code above to spit out clean data that the RL agent can work with
## define the states
## define the actions
## define the reward function
## define a function to identify the state the agent is
## define a function that lets the agent choose an action
## define the q-matrix and how to calculate it
## define the update the method 
## add exploration vs. exploitation rate instead of randomly choosing at step 1 then basing off q-value
## add more states based on other indicators (added MACD)
## record the actions and plot out how it would look like in terms of $$ gains
## change reward function from -1, 0 and 1 to delta of today_price- yesterday_price

"""TO BE COMPLETED"""
## add a decaying exploration function
    # when exploring, choose any action that is not highest q-value
## double check the reward and q_value calculation to make sure it works as intended
## add even more states based on other indicators
    # add MACD movement direction (if MACD,t=0 < avg(MACD,t={-3:-1}), then MACD movement direction is down)
    # add RSI movement direction (similar to MACD movement calculation above)
## add contraints (cant sell if there is no stock in holding)    


def state_action_pair():

    states = ['RSI<25 & MACD < 0', 'RSI<50 & MACD < 0', 'RSI<75 & MACD < 0', 'RSI<100 & MACD < 0',
              'RSI<25 & MACD > 0', 'RSI<50 & MACD > 0', 'RSI<75 & MACD > 0', 'RSI<100 & MACD > 0']
    
    q_table = pd.DataFrame(states).rename(columns={0:'states'})
    
    q_table['B'],q_table['S'],q_table['H'] = 0,0,0

    return q_table

q_table = state_action_pair()
q_table = q_table.set_index('states')    

def get_current_state(df, i):
    rsi = df.iloc[i]['RSI']
    macd = df.iloc[i]['macd_diff_signal']

    # RSI changes, macd is fixed
    if rsi < 25 and macd < 0:
        rsi_state = 'RSI<25 & MACD < 0'
    elif rsi < 50 and macd < 0:
        rsi_state = 'RSI<50 & MACD < 0'
    elif rsi < 75 and macd < 0:
        rsi_state = 'RSI<75 & MACD < 0'
    elif rsi < 100 and macd < 0:
        rsi_state = 'RSI<100 & MACD < 0'

    # MACD changes, RSI is fixed
    if rsi < 25 and macd > 0:
        rsi_state = 'RSI<25 & MACD > 0'
    elif rsi < 50 and macd > 0:
        rsi_state = 'RSI<50 & MACD > 0'
    elif rsi < 75 and macd > 0:
        rsi_state = 'RSI<75 & MACD > 0'
    elif rsi < 100 and macd > 0:
        rsi_state = 'RSI<100 & MACD > 0'
    else:
        print('rsi unknown')
        
    return rsi_state

# dummy function, using a linear decision between exploration vs. exploration for now
def decaying_exploration_rate(rsi_state):
    
    epsilon = 0.9
    decay = 0.99
    min_epsilon = 0.1
    for i in range(0,10):
        epsilon = max(min_epsilon, epsilon*decay)    
    
    return None

def choose_action(q_table, rsi_state, epsilon):
    
    # at first iteration, always randomly pick an action
    if q_table.loc[rsi_state]['B']==q_table.loc[rsi_state]['H']==q_table.loc[rsi_state]['S']:
        action_choice = random.choice(['B','H','S'])
        print('chosen randomly ', action_choice)
        action_choice_type = '1st run, random'
        
    # at any subsequent iteration, either explore (with p = eps) or exploit highest q-value (with p=1-eps)
    else:
        random_ = random.uniform(0,1)
        if  random_ < epsilon:
            max_value_choice = q_table.loc[rsi_state].idxmax(axis=1)
            action_list = ['B','H','S']
            refined_action_list = [value for value in action_list if value != max_value_choice]
            action_choice = random.choice(refined_action_list)
            action_choice_type = 'exploring, not 1st run'
            print('action ', action_choice, ' chosen randomly, random prob: ', random_, ' epsilon: ', epsilon)
        
        else:
            action_choice = q_table.loc[rsi_state].idxmax(axis=1)
            action_choice_type = ' q_value based'
            print('action ', action_choice, ' chosen based on q-value, random prob: ', random_, ' epsilon: ', epsilon)
    return action_choice, action_choice_type


# At EOD of the trading day, calculate the reward of the action taken at the beginnning of the day
def get_reward(df, i, j):
    recent_price = df[df.index==i]['detrended_price_adj'].values[0]
    old_price = df[df.index==j]['detrended_price_adj'].values[0]
    
    if recent_price > old_price:
        return recent_price - old_price
    elif recent_price == old_price:
        return 0
    elif recent_price < old_price:
        return recent_price - old_price

q_table = state_action_pair()
q_table = q_table.set_index('states')    

ALPHA = 0.8
GAMMA = 0.5

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', 500)

historical_action_table = pd.DataFrame(columns={'date', 'state', 'action', 'reward', 
                                                'price', 'action_choice', 'epsilon'})

eps = 0.9
    
for i in range(1,1055):
    
    j = i-1
    
    eps = eps*0.999

    if eps < 0.1:
        eps = 0.1
    
    reward = get_reward(df, i, j)

    rsi_state = get_current_state(df, i)
    print('the state is ', rsi_state)
    
    action_choice, action_choice_type = choose_action(q_table, rsi_state, eps)
    
    old_q_value = q_table.loc[rsi_state,action_choice]
    print('old q_value is ', old_q_value)
    
    new_q_value = old_q_value + ALPHA*(reward + GAMMA*old_q_value)
    print('new q_value is ', new_q_value)
    
    q_table.loc[rsi_state, action_choice] = new_q_value
    print(q_table)
    
    # fill in the historical action table analysis at a later time
    historical_action_table = historical_action_table.append({'date':str(df[df.index==i]['Date'].values[0])[:10],
                                                              'state':rsi_state, 'action':action_choice,
                                                              'reward':reward, 'price':df[df.index==i]['Close'].values[0],
                                                              'action_choice':action_choice_type, 'epsilon':eps}, 
                                                                ignore_index=True)

    print('--------------------------------------------')
    


    