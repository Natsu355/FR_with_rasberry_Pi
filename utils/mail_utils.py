import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


sender_email = "kratosgow1209@gmail.com"
sender_password = "werg zpbn glsp rvdy"
recipient_email = "reddykiran355@gmail.com"
subject = "Detected Face Alert"
body = "A face was detected. Please find the image attached."
#detected_face_image = "3-samp_saved.jpg"


def send_email_with_image(image_path):
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Add body text to the email
    msg.attach(MIMEText(body, 'plain'))

    # Open the image file and attach it to the email
    with open(image_path, 'rb') as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f'attachment; filename="{image_path}"')
        msg.attach(mime_base)

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print("------------ Email sent successfully! -----------")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()


#send_email_with_image(sender_email, sender_password, recipient_email, subject, body, detected_face_image)

