import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd, datetime as dt
import numpy as np

def get_data():
    historical_df = pd.read_csv('../unit_test/daily_prices.csv')
    
    return historical_df

def plot_macd_indicator(df):
    
    fig1, ax1 = plt.subplots()
    plt.xticks(rotation=45)

    color = 'tab:red'
    ax1.set_ylabel('Closing Stock Price', color=color)
    ax1.plot(df['date'], df['close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.xticks(np.arange(0, 365, step=20), rotation=45)
    
    color = 'tab:blue'
    ax2.set_ylabel('Relative Strength', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['rsi'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig1.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig1

def plot_rsi_indicator(df):
        
    fig1, ax1 = plt.subplots()
    plt.xticks(rotation=45)
    
    color = 'tab:red'
    ax1.set_ylabel('Closing Stock Price', color=color)
    ax1.plot(df['date'], df['close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.xticks(np.arange(0, 365, step=20), rotation=45)
    
    color = 'tab:blue'
    ax2.set_ylabel('MACD', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['date'], df['macd'], color='tab:green')
    ax2.plot(df['date'], df['signal_line'], color='tab:blue')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    return fig1

def save_as_pdf(df, ticker_name):
    today_date = str(dt.date.today())
    
    pdf = matplotlib.backends.backend_pdf.PdfPages('../daily_reports/'+ticker_name+'_'+today_date+'.pdf')
    pdf.savefig(plot_macd_indicator(df))
    pdf.savefig(plot_rsi_indicator(df))
    pdf.close()


