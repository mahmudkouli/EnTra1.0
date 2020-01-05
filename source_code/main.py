
import entra1_0
from email_engine import email_object
import indicator_plotter
import datetime, smtplib

def main():
    input_data = entra1_0.main()
    
    indicator_plotter.main()
    
    credentials = open('../utils/credentials.txt')
    credentials_list = credentials.read().split(',')
    credentials.close()

    user_name, psw = credentials_list[0], credentials_list[1]

    week_no = datetime.datetime.today().weekday()

    if week_no != 4 or week_no != 5:

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(user_name, psw)

        msg = email_object(input_data, user_name)
    
        s.send_message(msg)
        print('Successfully sent email')
        
        s.quit()
    
    indicator_plotter.remove_daily_reports()

main()

