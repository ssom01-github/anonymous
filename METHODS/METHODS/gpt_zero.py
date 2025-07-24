# -*- coding: utf-8 -*-
"""gptzero_mbart_hindi.py"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Hindi-supporting causal model
model_name = "facebook/mbart-large-50"  # Autoregressive, supports Hindi
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load your data
df = pd.read_csv("../HindiSumm/data/train.csv")  # Adjust path if needed

# Function to compute perplexity
def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        try:
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            return torch.exp(loss).item() if loss is not None else float('inf')
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

# GPTZero-like classification function
def classify_text(perplexity):
    if perplexity < 60:  # Placeholder thresholds; tune these
        return 0, "पाठ AI द्वारा उत्पन्न किया गया है।"
    elif perplexity < 70:
        return 0, "पाठ में संभवतः AI द्वारा उत्पन्न हिस्से हैं।"
    else:
        return 1, "पाठ मानव द्वारा लिखा गया है।"

# Process the dataframe
def process_gptzero(input_df, text_column='text', true_label_column='label'):
    # Compute perplexity
    input_df['perplexity'] = input_df[text_column].apply(calculate_perplexity)

    # Handle infinite values
    input_df['perplexity'] = input_df['perplexity'].replace([float('inf'), float('-inf')], np.nan)
    df_cleaned = input_df.dropna(subset=['perplexity'])

    # Classify based on perplexity
    classifications = df_cleaned['perplexity'].apply(classify_text)
    df_cleaned['predict_label'] = classifications.apply(lambda x: x[0])
    df_cleaned['classification_text'] = classifications.apply(lambda x: x[1])

    # Metrics if true labels exist
    if true_label_column in df_cleaned.columns:
        true_labels = df_cleaned[true_label_column].tolist()
        predicted_labels = df_cleaned['predict_label'].tolist()
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, zero_division=0))
    else:
        print(f"Warning: '{true_label_column}' not found. Skipping metrics.")

    # Print average perplexity
    print("\nAverage Perplexity:")
    print(f"Text: {df_cleaned['perplexity'].mean():.2f}")

    return df_cleaned

# Run the GPTZero-like process
text_column = 'text'  # Adjust based on your CSV column name
true_label_column = 'label'  # Adjust if ground truth exists
df_processed = process_gptzero(df, text_column, true_label_column)

# Save results
output_file = "../HindiSumm/data/train_predictions_gptzero_mbart_2.csv"
df_processed.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")