import mailbox
import os
import argparse
import html
from email.header import decode_header

def clean_header(header_value):
    """Decodes headers like Subject or From to handle special characters."""
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    header_str = ""
    for bytes_part, encoding in decoded_parts:
        if isinstance(bytes_part, bytes):
            try:
                header_str += bytes_part.decode(encoding or 'utf-8', errors='replace')
            except LookupError:
                header_str += bytes_part.decode('utf-8', errors='replace')
        else:
            header_str += str(bytes_part)
    return header_str

def decode_part_payload(part):
    """Decodes a message part's payload, returning None on failure."""
    try:
        charset = part.get_content_charset() or 'utf-8'
        payload = part.get_payload(decode=True)
        if not payload:
            return None
        return payload.decode(charset, errors='replace')
    except Exception:
        return None

def extract_multipart_content(message):
    """Extracts HTML and text content from multipart message."""
    html_body = None
    text_body = None
    
    for part in message.walk():
        ctype = part.get_content_type()
        cdispo = str(part.get('Content-Disposition'))
        
        if 'attachment' in cdispo:
            continue
        
        decoded_payload = decode_part_payload(part)
        if not decoded_payload:
            continue
        
        if ctype == 'text/html':
            return decoded_payload, text_body
        elif ctype == 'text/plain':
            text_body = decoded_payload
    
    return html_body, text_body

def extract_simple_content(message):
    """Extracts HTML and text content from non-multipart message."""
    ctype = message.get_content_type()
    decoded = decode_part_payload(message)
    
    if not decoded:
        return None, None
    
    if ctype == 'text/html':
        return decoded, None
    elif ctype == 'text/plain':
        return None, decoded
    
    return None, None

def format_content(html_body, text_body):
    """Formats the final content, converting text to HTML if needed."""
    if html_body:
        return html_body
    
    if text_body:
        escaped_text = html.escape(text_body)
        return f"<div style='font-family: sans-serif; white-space: pre-wrap;'>{escaped_text}</div>"
    
    return "<i>[No content found]</i>"

def get_email_content(message):
    """
    Extracts content preferring HTML. 
    If only plain text exists, converts it to valid HTML.
    """
    if message.is_multipart():
        html_body, text_body = extract_multipart_content(message)
    else:
        html_body, text_body = extract_simple_content(message)
    
    return format_content(html_body, text_body)

def split_mbox_to_html(mbox_path, output_dir):
    if not os.path.isfile(mbox_path):
        print(f"Error: The file '{mbox_path}' was not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing '{mbox_path}'...")
    mbox = mailbox.mbox(mbox_path)
    
    count = 0

    for i, message in enumerate(mbox):
        try:
            # 1. Extract Headers
            subject = clean_header(message['subject']) or "(No Subject)"
            sender = clean_header(message['from']) or "(Unknown Sender)"
            date = clean_header(message['date']) or "(No Date)"
            
            # 2. Get Body (HTML or Text-converted-to-HTML)
            body_content = get_email_content(message)

            # 3. Create the HTML Page Structure
            # We add a gray header block at the top for metadata
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{html.escape(subject)}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                    .email-container {{ background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }}
                    .header {{ border-bottom: 1px solid #ddd; padding-bottom: 15px; margin-bottom: 20px; color: #555; }}
                    .header div {{ margin-bottom: 5px; }}
                    .label {{ font-weight: bold; color: #333; }}
                    .content {{ line-height: 1.6; color: #000; }}
                </style>
            </head>
            <body>
                <div class="email-container">
                    <div class="header">
                        <div><span class="label">Subject:</span> {html.escape(subject)}</div>
                        <div><span class="label">From:</span> {html.escape(sender)}</div>
                        <div><span class="label">Date:</span> {html.escape(date)}</div>
                    </div>
                    <div class="content">
                        {body_content}
                    </div>
                </div>
            </body>
            </html>
            """

            # 4. Save to file
            safe_subject = "".join([c if c.isalnum() else "_" for c in subject])[:50]
            filename = os.path.join(output_dir, f"{i+1:04d}_{safe_subject}.html")

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_template)

            count += 1
            if count % 100 == 0:
                print(f"Saved {count} emails...", end='\r')

        except Exception as e:
            print(f"\nError processing email #{i+1}: {e}")

    print(f"\nDone! Saved {count} HTML files to '{output_dir}/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mbox emails to individual .html files.")
    parser.add_argument("input_file", help="Path to the input .mbox file")
    parser.add_argument("output_dir", help="Directory to save the html files")
    
    args = parser.parse_args()
    
    split_mbox_to_html(args.input_file, args.output_dir)