from bs4 import BeautifulSoup
import re
import pandas as pd
import tqdm

html_content = ""
for i in tqdm.tqdm(range(1, 17), desc="Loading HTML files"):
    filename = f'message_{i}.html'
    with open(filename, 'r', encoding='utf-8') as file:
        html_content += file.read()

soup = BeautifulSoup(html_content, 'html.parser')

messages = soup.find_all('div', class_='_a6-g')

last_user_message = None
last_assistant_message = None
last_role = None

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F700-\U0001F77F"
                           u"\U0001F780-\U0001F7FF"
                           u"\U0001F800-\U0001F8FF" 
                           u"\U0001F900-\U0001F9FF" 
                           u"\U0001FA00-\U0001FA6F" 
                           u"\U0001FA70-\U0001FAFF" 
                           u"\U00002702-\U000027B0"  
                           u"\U000024C2-\U0001F251"  
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

dataset = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]
}

for message in tqdm.tqdm(messages, desc="Processing messages with BeautifulSoup"):
    sender = message.find('div', class_='_2ph_ _a6-h _a6-i').text.strip()
    content = message.find('div', class_='_2ph_ _a6-p').text.strip()
    
    content = remove_emojis(content)
    
    if not content:
        continue
    
    if sender == 'sendername':
        role = 'user'
    elif sender == 'sendername':
        role = 'assistant'
    else:
        continue
    
    if content in ["Kontakt zmeškal vaše volání.", "Kontakt zmeškal váš videohovor.", "Pro video klikněte:", "Zmeškali jste hovor od kontaktu."]:
        continue
    
    if any(phrase in content for phrase in ["Videohovor skončil", "Volal vám uživatel Messengeru", "Uživatel Messengeru zmeškal váš hovor.", "Máte zameškaný hovor od uživatele Messengeru"]):
        continue
    
    if role == 'user':
        if last_role != 'user':
            last_user_message = content
            dataset["messages"].append({"role": role, "content": content})
            last_role = 'user'
    elif role == 'assistant':
        if last_role != 'assistant':
            last_assistant_message = content
            dataset["messages"].append({"role": role, "content": content})
            last_role = 'assistant'

df = pd.DataFrame(dataset["messages"])

with tqdm.tqdm(total=1, desc="Saving dataset") as pbar:
    df.to_json("dataset22.json", orient="records", lines=True)
    pbar.update(1)
