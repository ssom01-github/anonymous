import pandas as pd
import re
from indicnlp.tokenize import indic_tokenize

# === Utility Functions ===

def preprocess_text(text):
    """
    Removes the title enclosed in ** ** and returns the rest.
    """
    match = re.match(r"\*\*(.*?)\*\*", text)
    if match:
        text = re.sub(r"\*\*.*?\*\*", "", text, count=1).strip()
    return text

def remove_first_sentence_if_colon(text):
    """
    Removes the first sentence if it ends with a colon (:).
    """
    pattern = r'^[^।!?]*:'
    cleaned_text = re.sub(pattern, '', text, count=1).strip()
    return cleaned_text

def preprocess_normal(text):
    """
    Removes the first sentence based on Hindi '।' or pipe '|'.
    """
    match = re.search(r'[।|]', text)
    if match:
        return text[match.end():].strip()
    return text.strip()

def remove_starting_english(text):
    """
    Removes any leading English content or metadata until the first Devanagari character.
    """
    match = re.search(r'[ऀ-ॿ]', text)
    if match:
        return text[match.start():].strip()
    return text.strip()

def is_mostly_english(text, threshold=0.8):
    """
    Checks if a large proportion of characters in the text are English.
    """
    if not text:
        return True
    total_chars = len(text)
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    return (english_chars / total_chars) > threshold


# === Main Formatter ===

def format_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna().reset_index(drop=True)

    original_articles = data['original_article']
    generated_articles = data['generated_article']
    # keywords = data['keywords']
    summary = data['summary']

    new_data = pd.DataFrame(columns=['id', 'text', 'label', 'summary'])

    for i in range(len(original_articles)):
        # Clean both original and generated articles
        gen_article = remove_starting_english(generated_articles[i])
        orig_article = preprocess_normal(original_articles[i])
        orig_article = remove_first_sentence_if_colon(orig_article)

        # Skip if generated article is mostly English
        if is_mostly_english(gen_article):
            continue

        # Append both real and fake samples
        new_data = new_data._append({'id': i, 'text': orig_article, 'label': 1, 'keywords': summary[i]}, ignore_index=True)
        new_data = new_data._append({'id': i, 'text': gen_article, 'label': 0, 'keywords': summary[i]}, ignore_index=True)

    return new_data


# === Run Formatter ===

# data_path = './data/hindi_generated_data_llama3.csv'
data_path = './Generated_Articles_From_Summary_llama_1000.csv'
new_data = format_data(data_path)

# Save cleaned output
new_data.to_csv('./data/Data_llama3.csv', index=False)
print("Cleaned data saved to ./data/Data_llama3.csv")
