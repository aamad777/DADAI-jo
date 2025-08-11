import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

sender = os.getenv("GMAIL_USER")
password = os.getenv("GMAIL_PASS")
receiver = os.getenv("DAD_EMAIL")

try:
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(sender, password)
        subject = "Test Email"
        body = "This is a test from Ask Dad AI."
        message = f"Subject: {subject}\n\n{body}"
        smtp.sendmail(sender, receiver, message)
    print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Email failed:", e)
