import pandas as pd
import os
import google.generativeai as genai
from keybert import KeyBERT
from google.cloud import language_v1
import stanza
import yake
import time
import re

import nltk
from nltk.tag import tnt
from nltk.corpus import indian
from nltk.tree import Tree

# Configure the Gemini API
os.environ['GOOGLE_API_KEY'] = "AIzaSyAaclfgYAqOR0eF2juAUErLofXpUI6cwjM"  # Replace with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])



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



# Filter for real news articles
def filter_real_articles(df):
    """Filter rows where the label is 1 (real news)."""
    return df[df['label'] == 1]

# Extract keywords using KeyBERT
def extract_keywords_with_keybert(text, num_keywords=10):
    """
    Extract keywords from the text using KeyBERT.
    :param text: The input text to analyze.
    :param num_keywords: The number of keywords to extract.
    :return: A string of comma-separated keywords.
    """
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words='english', top_n=num_keywords)
    return ", ".join([kw[0] for kw in keywords])

# Extract named entities using Stanza
def extract_named_entities_stanza(text):
    """
    Extract named entities from the text using Stanza.

    :param text: The input text to analyze.
    :return: A string of comma-separated named entities.
    """
    doc = nlp(text)
    entities = [ent.text for sent in doc.sentences for ent in sent.ents]
    return ", ".join(entities)

def extract_keywords(text,num_keywords=10):
    """
    Extract keywords using Yake.
    """
    language = "hi"  # Specify language as Hindi
    max_ngram_size = 2
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)
    keywords = custom_kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

# Generate a new article using Gemini API
def generate_article(keywords, num_words,language="Hindi"):
    """
    Generate an article using Gemini's content generation.
    :param keywords: Keywords extracted from the article text for context.
    :param language: Language of the output article.
    """
    prompt = f"""
        You are an assitant skilled in writing {language} wiki articles. Expand the following keywords into a meaningful {language} article of approximately {num_words-30}-{num_words+30} words. Keywords: "{keywords}".
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
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
    # Paths and configuration
    model = hindi_model()

    input_file = "./data/hindi_qa_analysis.csv"  # Path to your CSV file
    # output_file = "Generated_articles_gemini_key_nltk.csv"  # Path to save the new articles
    output_file = "./data/hindi_generated_data.csv"  # Path to save the new articles
    # Load and filter dataset
    df = pd.read_csv(input_file)
    real_articles = df['context'] 
    # remove any duplicate articles
    real_articles = real_articles.drop_duplicates()
    print(f"Number of real articles: {len(real_articles)}")
       
    output_df = pd.DataFrame(columns=['keywords', 'original_article', 'generated_article'])
    cnt = 0

    # Generate articles for each real article
    for text in real_articles:
        try:
            keywords = extract_keywords(text, num_keywords=15)
            # keywords = get_keywords_nltk(model.tag(nltk.word_tokenize(text)))
            print(f"Extracted keywords: {keywords}")
            # Skip articles with no keywords
            if not keywords:
                print("No keywords extracted. Skipping article.")
            # continue
            # Generate a new article
            new_article = generate_article(keywords, num_words=130) # should be 130 but initall used 230
            # new_article = "PPP"
            # Append results to the list
            output_df = output_df._append({
                'keywords': ", ".join(keywords),
                
                'original_article': text,
                'generated_article': new_article
            }, ignore_index=True)
            cnt += 1
            if cnt == 5:
                print("# Sleeping for 40 seconds to avoid rate limiting...")
                time.sleep(1)
                cnt = 0
                
        except Exception as e:
            print(f"Error processing article: {e}")

    # Save the generated articles to a new CSV file
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Generated articles saved to {output_file}")
    
    # #create a new df
    # newdf = pd.DataFrame(columns=['id','text','label'])
    # # for real articles add label 1
    # for i in range(len(output_df)):
    #     newdf = newdf._append({'id':i,'text':preprocess_text(output_df['generated_article'][i]),'label':0},ignore_index=True)
    #     newdf = newdf._append({'id':i,'text':preprocess_text(output_df['original_article'][i]),'label':1},ignore_index=True)
    # print(f"Generated articles saved to hindi_analysis.csv")
    # newdf.to_csv("./data/Data_gemini.csv",index=False)

if __name__ == "__main__":
    main()
