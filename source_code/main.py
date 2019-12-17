
import entra1_0
from email_engine import email_object
import datetime, email, smtplib

def main():
	input_data = entra1_0.main()

	email = 'entra.daily@gmail.com'

	credentials = open('../utils/credentials.txt')
	credentials_list = credentials.read().split(',')
	credentials.close()

	user_name, psw = credentials_list[0], credentials_list[1]

	week_no = datetime.datetime.today().weekday()

	if week_no != 5 or week_no != 4:
		s = smtplib.SMTP('smtp.gmail.com', 587)
		s.starttls()
		s.login(user_name, psw)

		msg = email_object(input_data, user_name)

		s.send_message(msg)
		print('Successfully sent email')

		s.quit()

	else:
	    pass

	return input_data.to_csv('main.csv')

main()