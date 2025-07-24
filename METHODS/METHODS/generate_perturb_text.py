

# # Example Usage
# # augment_dataset("./data/sample.csv", "./data/sample_perturb.csv", percentage=0.2)
# augment_dataset("./data/Data_gemini_2k.csv", "./data/Data_gemini_2k_perturb.csv", percentage=0.2)

import pandas as pd
import random
import stanza
from tqdm.auto import tqdm
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph import unsupervised_morph 
from nltk.corpus import wordnet as wn
import nltk
import torch

# Download necessary files
stanza.download('hi')  # Hindi NLP model
nltk.download('omw-1.4')  # Required for WordNet
nltk.download('wordnet')  # WordNet
nlp = stanza.Pipeline('hi')  # Load Hindi NLP pipeline

from pyiwn.iwn import IndoWordNet,Language
iwn = IndoWordNet(lang=Language.HINDI)  # Load Hindi WordNet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_hindi_synonyms(word):
    """
    Get Hindi synonyms using IndoWordNet (`pyiwn`).
    """
    synonyms = set()
    try:
        synsets = iwn.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
    except:
        return []
    
    synonyms.discard(word)  # Remove original word
    return list(synonyms)

def process_text_with_pos(text):
    """
    Tokenizes and assigns POS tags to words in a Hindi text.
    Returns a list of (word, POS) tuples.
    """
    doc = nlp(text)  # Process text with Stanza
    words_with_pos = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            words_with_pos.append((word.text, word.upos))  # Extract word & POS tag
    
    return words_with_pos

def synonym_replacement(words_with_pos, n):
    """
    Replaces `n` words (only NOUN, VERB, ADJ, ADV) in the sentence with synonyms.
    """
    new_words = [word[0] for word in words_with_pos]  # Extract original words
    target_words = [word for word in words_with_pos if word[1] in ["NOUN", "VERB", "ADJ", "ADV"]]

    if not target_words:
        return " ".join(new_words)  # No replacements if no eligible words

    random.shuffle(target_words)
    num_replaced = 0

    for word, pos in target_words:
        synonyms = get_hindi_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [str(synonym) if w == word else w for w in new_words]
            num_replaced += 1

        if num_replaced >= n:
            break

    return " ".join(new_words)

def augment_dataset(input_csv, output_csv, percentage=0.2):
    """
    Augment dataset by replacing some Hindi nouns, verbs, adjectives, and adverbs with synonyms.
    """
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    if not {"id", "text", "label", "summary"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'id', 'text', 'label', 'summary' columns")

    augmented_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = str(row["text"])  # Convert to string (handle NaN cases)
        doc = nlp(text)  # Process text with Stanza
        sentences = [sentence.text for sentence in doc.sentences]  # Extract sentences

        new_sentences = []

        for sent in sentences:
            words_with_pos = process_text_with_pos(sent)
            num_replace = int(len(words_with_pos) * percentage)  # Calculate words to replace
            new_sentences.append(synonym_replacement(words_with_pos, num_replace))

        augmented_text = " ".join(new_sentences)
        augmented_data.append([row["id"], text, augmented_text, row["label"], row["summary"]])

    # Create new DataFrame
    augmented_df = pd.DataFrame(augmented_data, columns=["id", "text", "text_perturb", "label", "summary"])
    
    # Save to output CSV
    augmented_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Augmented dataset saved to {output_csv}")

# Example Usage
# augment_dataset("./data/train.csv", "./data/train_perturb.csv", percentage=0.2)
augment_dataset("./data/test.csv", "./data/test_perturb.csv", percentage=0.2)
augment_dataset("./data/val.csv", "./data/val_perturb.csv", percentage=0.2)
# augment_dataset("./data/sample.csv", "./data/sample_perturb.csv", percentage=0.2)
# augment_dataset("./data/Data_gemini_2k.csv", "./data/Data_gemini_2k_perturb.csv", percentage=0.2)