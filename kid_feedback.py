# kid_feedback.py
import os
import smtplib
import ssl
from email.message import EmailMessage

def _cfg(key, *alts, default=None):
    for k in (key, *alts):
        v = os.getenv(k)
        if v:
            return v
    return default

def _smtp_config_summary():
    return (
        f"SMTP_HOST={_cfg('SMTP_HOST', default='smtp.gmail.com')}\n"
        f"SMTP_PORT={_cfg('SMTP_PORT', default='587')}\n"
        f"EMAIL_FROM={_cfg('EMAIL_FROM', default=_cfg('SMTP_USER','GMAIL_USER','EMAIL_USER'))}\n"
        f"EMAIL_TO={_cfg('EMAIL_TO','DAD_EMAIL_TO')}\n"
    )

def send_email_to_dad(child_name: str, question: str, answer: str, to_email: str | None = None):
    """
    Sends an email to Dad with the child's question and the current answer.
    Returns: (ok: bool, message: str)
    .env example (Gmail + App Password):
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USER=yourgmail@gmail.com
        SMTP_PASS=your_gmail_app_password
        EMAIL_FROM=yourgmail@gmail.com
        EMAIL_TO=dad@family.com
    """
    smtp_host = _cfg("SMTP_HOST", default="smtp.gmail.com")
    smtp_port = int(_cfg("SMTP_PORT", default="587"))
    smtp_user = _cfg("SMTP_USER", "GMAIL_USER", "EMAIL_USER")
    smtp_pass = _cfg("SMTP_PASS", "GMAIL_APP_PASSWORD", "EMAIL_PASS")
    email_from = _cfg("EMAIL_FROM", default=smtp_user)
    email_to = to_email or _cfg("EMAIL_TO", "DAD_EMAIL_TO")

    if not smtp_user or not smtp_pass or not email_to:
        return (False,
            "Email not configured.\n"
            "Set SMTP_USER, SMTP_PASS, and EMAIL_TO in your .env "
            "(or GMAIL_USER/GMAIL_APP_PASSWORD/DAD_EMAIL_TO).\n\n"
            "Current config:\n" + _smtp_config_summary()
        )

    subject = f"[Ask Dad] {child_name} needs help"
    body = (
        f"Child Name: {child_name}\n"
        f"Question: {question}\n"
        f"Current Answer: {answer}\n"
    )

    msg = EmailMessage()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if smtp_port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=20) as s:
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
                s.ehlo()
                s.starttls(context=context)
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)

        return (True, f"Sent to {email_to} via {smtp_host}:{smtp_port}")
    except smtplib.SMTPAuthenticationError as e:
        return (False, "Authentication failed (check SMTP_USER/SMTP_PASS). "
                       "If using Gmail, you MUST use an App Password.\n\n" + str(e))
    except smtplib.SMTPConnectError as e:
        return (False, f"Connection failed to {smtp_host}:{smtp_port}. Is the port blocked?\n{e}")
    except Exception as e:
        return (False, f"Send failed: {e}\n\nCurrent config:\n" + _smtp_config_summary())
