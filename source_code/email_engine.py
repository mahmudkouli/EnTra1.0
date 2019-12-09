import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def email_object(data, user_name):
	msg = MIMEMultipart()

	msg['From']= user_name
	msg['To']= 'mahmudkouli@gmail.com'
	#msg['CC']='ksenia.ter@gmail.com'
	msg['Subject']="TESTING: Entra Alert, " + str(datetime.date.today())

	msg.attach(MIMEText(data.to_html(), 'html'))

	return msg
