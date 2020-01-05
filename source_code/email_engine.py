import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os 
from email.mime.base import MIMEBase
from email import encoders

def email_object(data, user_name):
    msg = MIMEMultipart()
    
    attachments = os.listdir('../daily_reports')
    
    # open the file to be sent
    
    for file in attachments:
            
        print(file)
        attachment = open('../daily_reports/'+file, "rb")
    
        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')
        
        # To change the payload into encoded form
        p.set_payload((attachment).read())
        
        # encode into base64
        encoders.encode_base64(p)
        
        p.add_header('Content-Disposition', "attachment; filename= %s" % file)
        
        # attach the instance 'p' to instance 'msg'        
        msg.attach(p)

    msg['From']= user_name
    msg['To']= 'mahmudkouli@gmail.com'
    #msg['CC']='ksenia.ter@gmail.com'
    msg['Subject']="TESTING: Entra Alert, " + str(datetime.date.today())

    msg.attach(MIMEText(data.to_html(), 'html'))

    return msg




