import os.path
import pickle
import base64
import json
import re

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    #use your own path
    #use your own path
    #use your own path
    if os.path.exists('LLM/token.pickle'):
        with open('LLM/token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                'LLM/client_secret_349717392152-a31crp53p3kkaq5imdm5q41vdfka5e7n.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('LLM/token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('gmail', 'v1', credentials=creds)
    print("Gmail Service Created")
    return service

def list_messages(service, user_id='me', query='in:inbox'):
    """
    List all messages in the user's inbox.
    """
    messages = []
    results = service.users().messages().list(userId=user_id, q=query).execute()
    if 'messages' in results:
        messages.extend(results['messages'])
    #Next page
    while 'nextPageToken' in results:
        page_token = results['nextPageToken']
        results = service.users().messages().list(userId=user_id, q=query, pageToken=page_token).execute()
        if 'messages' in results:
            messages.extend(results['messages'])
    return messages

def get_message(service, msg_id, user_id='me'):
    """
    Get a message by id.
    """
    message = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()
    return message

def parse_email_headers(message):
    """
    Parse email headers.
    """
    headers = message.get('payload', {}).get('headers', [])
    email_info = {}
    for header in headers:
        name = header.get('name')
        value = header.get('value')
        if name in ['From', 'Subject', 'Date']:
            email_info[name] = value
    return email_info

def get_email_body(message):
    """
    Get email body.
    """
    parts = message.get('payload', {}).get('parts', [])
    body = ''
    if parts:
        for part in parts:
            mime_type = part.get('mimeType')
            if mime_type == 'text/plain':
                data = part.get('body', {}).get('data')
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                    break
    else:
        data = message.get('payload', {}).get('body', {}).get('data')
        if data:
            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
    return body

def remove_urls(text):
    """
    Remove URLs from text.
    """
    url_pattern = re.compile(r'http[s]?://\S+')
    return url_pattern.sub('', text)

def process_all_emails(service):
    # List all messages in the user's inbox
    messages = list_messages(service, query='in:inbox')
    print(f"Find {len(messages)} Mails")
    
    all_emails = []
    
    for msg_meta in messages:
        try:
            msg = get_message(service, msg_meta['id'])
            email_info = parse_email_headers(msg)
            email_body = get_email_body(msg)
            email_body_clean = remove_urls(email_body).strip()
            email_data = {
                "From": email_info.get('From', ''),
                "Subject": email_info.get('Subject', ''),
                "Date": email_info.get('Date', ''),
                "Content": email_body_clean
            }
            all_emails.append(email_data)
        except Exception as e:
            print(f"Emial ID {msg_meta['id']} Error: {e}")
    
    return all_emails

if __name__ == '__main__':
    service = get_gmail_service()
    emails = process_all_emails(service)
    
    with open('all_emails.json', 'w', encoding='utf-8') as f:
        json.dump(emails, f, ensure_ascii=False, indent=4)
    
    print("SAVE all_emails.json")