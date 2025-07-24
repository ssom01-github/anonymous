import pandas as pd 
import os
import pandas as pd
import re

# Function to process the text
def preprocess_text(text):
    match = re.match(r"\*\*(.*?)\*\*", text)  # Find text within ** **
    if match:
        title = match.group(1)  # Extract title
        text = re.sub(r"\*\*.*?\*\*", "", text, count=1).strip()  # Remove the first **...**
        return f"{text}"
    
import re

def remove_marked_sentence(text):
    """
    Removes a sentence that starts with '##' and ends with '|', '!', or ':'.
    
    Args:
    text (str): The input text.
    
    Returns:
    str: The processed text with the marked sentence removed.
    """
    # Regular expression pattern to match the marked sentence
    pattern = r'##[^|!:]*[|!:]'
    cleaned_text = re.sub(pattern, '', text).strip()
    
    return cleaned_text

import re

def remove_first_sentence_if_colon(text):
    """
    Removes the first sentence if it ends with a colon (:).
    
    Args:
    text (str): The input text.
    
    Returns:
    str: The processed text with the first sentence removed if it ends with ':'.
    """
    # Match the first sentence that ends with ':'
    pattern = r'^[^।!?]*:'
    cleaned_text = re.sub(pattern, '', text, count=1).strip()
    
    return cleaned_text



    
def process_title_lines(text):
    """
    Processes a multi-line text, extracting the 'text' part from lines that start with '##title : text'.

    Args:
        text (str): The input multi-line text.

    Returns:
        str: A string containing the extracted 'text' parts, separated by newlines.
    """
    lines = text.splitlines()
    extracted_texts = []

    for line in lines:
        match = re.match(r"^##.*?:\s*(.*)$", line)
        if match:
            extracted_texts.append(match.group(1).strip())

    return "\n".join(extracted_texts)

def preprocess_normal(text):
    # Split the text at the first occurrence of '।' or '|'
    match = re.search(r'[।|]', text)
    if match:
        return text[match.end():].strip()  # Return the text after the first full stop
    return text.strip()  # If no full stop found, return the original text



def format_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    data = data.reset_index(drop=True)
    # labels : summary,original_article,__ORIGINAL_ARTICLE_END__,generated_article,__GENERATED_ARTICLE_END__
    original_articles = data['original_article']
    generated_articles = data['generated_article']
    summaries = data['summary']
    
    # remove new line characters from the generated articles
    generated_articles = [article.replace('\n',' ') for article in generated_articles]
    # first line contains a sentence iwth **text**
    # remove them from the generated articles to put text: at the start
    generated_articles = [remove_marked_sentence(article) for article in generated_articles]    
    # generated_articles = [preprocess_text(article) for article in generated_articles] 
    original_articles = [preprocess_normal(article) for article in original_articles]
    original_articles = [remove_first_sentence_if_colon(article) for article in original_articles]
    
    # create a new dataframe
    new_data = pd.DataFrame(columns=['id','text','label','summary'])
    for i in range(len(original_articles)):
        new_data = new_data._append({'id':i,'text':original_articles[i],'label':1,'summary':summaries[i]},ignore_index=True)
        new_data = new_data._append({'id':i,'text':generated_articles[i],'label':0,'summary': summaries[i]},ignore_index=True)
    return new_data

data_path = './Generated_Articles_From_Summary_gemini_1000.csv'
new_data = format_data(data_path)
new_data.to_csv('./Data_gemini_2k.csv',index=False)
print("Data saved to Data_gemini_2k.csv")
