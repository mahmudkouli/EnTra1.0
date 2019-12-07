import email, smtplib, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd

def data_to_send():
    df = pd.read_csv('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/daily_updated_prices.csv').round({'close':2, 'rsi':2, 'macd_diff_signal':2})
    stocks = pd.read_excel('/Users/Mahmud/Desktop/stock_market_tracker/prod_env/EnTra/ticker_data.xlsx')
    merged = pd.merge(df,stocks,how='left',left_on = 'ticker', right_on='Ticker')
    merged = merged[['ticker', 'Company Name', 'Industry', 'date', 'close', 'rsi', 'macd_diff_signal','daily mean volume', 'action_signal']]
    merged = merged.rename(columns = {'ticker':'Ticker', 'date':'Date', 'close':'Close', 'rsi':'RSI', 'macd_diff_signal':'MACD', 'action_signal':'Action'})


    daily_df = merged[(merged['Date'] == merged.Date.max()) & 
                      (merged['RSI']<35) & 
                      (merged['Close']>10) & 
                      (merged['daily mean volume'] > 500000) &
                      (merged['Action']!='No signal')].sort_values(by=['RSI','MACD']) 
                      
    daily_df = daily_df[['Ticker', 'Company Name', 'Industry', 'Date', 'Close', 'RSI', 
                         'MACD','daily mean volume','Action']]
    return daily_df

def email_connection(email, msg, data):

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('XYZ_user@gmail.com', 'XYZ_password')
    msg['From']= email
    msg['To']= 'mahmudkouli@gmail.com'
    #msg['CC']='ksenia.ter@gmail.com'
    msg['Subject']="TESTING: Entra Alert, " + str(datetime.date.today())

    msg.attach(MIMEText(data.to_html(), 'html'))

    s.send_message(msg)
    print('Successfully sent email')

    s.quit()

def main():
    email = 'entra.daily@gmail.com'
    msg = MIMEMultipart()

    data = data_to_send()[['Ticker', 'Company Name', 'Industry', 'Date', 'Close', 'RSI', 'MACD','Action']]
    week_no = datetime.datetime.today().weekday()

    if week_no != 5 or week_no != 4:
        email_connection(email, msg, data)
    else:
        pass

if __name__ == "__main__":
    main()