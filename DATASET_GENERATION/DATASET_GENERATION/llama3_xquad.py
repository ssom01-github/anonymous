import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from keybert import KeyBERT
import yake
import time
import re

import nltk
from nltk.tag import tnt
from nltk.corpus import indian
from nltk.tree import Tree

# Configuration for Llama-3 model
MODEL_NAME = "meta-llama/Llama-3.2-3B-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AVG_LENGTH = 230  # Adjust length as needed
NUM_ARTICLES = 1000
INPUT_FILE = "./data/hindi_qa_analysis.csv"  # Path to your CSV file
OUTPUT_FILE = "./data/hindi_generated_data_llama3.csv"  # Path to save the new articles

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

def hindi_model():
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger

def get_keywords_nltk(pos, num_keywords=5):
    grammar = r"""NP:{<NN.*>}"""
    chunkParser = nltk.RegexpParser(grammar)
    chunked = chunkParser.parse(pos)
    continuous_chunk = set()
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.add(named_entity)
                current_chunk = []
            else:
                continue
    return list(continuous_chunk)[:num_keywords]

def extract_keywords(text, num_keywords=10):
    """
    Extract keywords using Yake.
    """
    language = "hi"  # Specify language as Hindi
    max_ngram_size = 2
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)
    keywords = custom_kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def generate_article(keywords, avg_length, language="Hindi"):
    """
    Generate a detailed article in the specified language from the given title and keywords.
    """
    prompt = (
        f"You are an assistant skilled in writing {language} language articles. "
        f"Using the keywords provided, generate a detailed article of approximately "
        f"{avg_length} words.\n\nKeywords: {keywords}\n\n"
        f"Please make sure the article is detailed and close to {avg_length} words, but do not exceed "
        f"{avg_length + 100} words."
    )

    try:
        # Tokenize and generate the article
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=int(avg_length * 1.5),
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode the generated output
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()  # Remove the prompt from the output

        # Clean and format the response
        return ' '.join(response.replace("\n", " ").split())

    except Exception as e:
        print(f"Error generating article: {e}")
        return None

def preprocess_text(text):
    """
    Preprocess the text by removing new lines and extra spaces.
    """
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Main function
def main():
    # Load and filter dataset
    start_time =  time.time()
    df = pd.read_csv(INPUT_FILE)
    real_articles = df['context']  # Assuming 'context' column contains the articles
    real_articles = real_articles.drop_duplicates()
    print(f"Number of real articles: {len(real_articles)}")
       
    output_df = pd.DataFrame(columns=['keywords', 'original_article', 'generated_article'])
    cnt = 0

    # Generate articles for each real article
    for idx, text in enumerate(real_articles):
        try:
            # Assume 'title' and 'keywords' are columns in your CSV dataset
            print(f"Processing article {idx+1}/{len(real_articles)}")

            # Extract keywords if necessary
            extracted_keywords = extract_keywords(text, num_keywords=15)
            print(f"Extracted keywords: {extracted_keywords}")

            # Generate a new article using Llama-3
            new_article = generate_article(extracted_keywords, AVG_LENGTH)

            # Append results to the output dataframe
            output_df
            output_df = output_df._append({
                'keywords': ", ".join(extracted_keywords),
                'original_article': text,
                'generated_article': new_article
            }, ignore_index=True)

            cnt += 1
            if cnt == 5:
                print("# Sleeping for 40 seconds to avoid rate limiting...")
                time.sleep(40)  # Adjust sleep time as needed
                cnt = 0

        except Exception as e:
            print(f"Error processing article {idx+1}: {e}")

    # Save the generated articles to a new CSV file
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Generated articles saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
