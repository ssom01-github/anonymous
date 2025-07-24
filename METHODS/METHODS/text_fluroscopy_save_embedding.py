import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import os

# Load IndicBERT tokenizer and model
model_name = "ai4bharat/indic-bert"
# use Bert-base-uncased for now
# distilbert-base-uncased
# model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Paths for data
train_path = '../../HindiSumm/data/train.csv'
save_path = './embeddings/'
os.makedirs(save_path, exist_ok=True)

# Load Hindi dataset
df = pd.read_csv(train_path)
# drop rows with empty text
df = df.dropna(subset=['text'])
# 500
df = df.sample(n=500, random_state=42)
print("Dataset loaded successfully!")


# Extract embeddings
def extract_embeddings(texts, labels, save_file):
    embeddings = []

    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            text_embedding = torch.stack(hidden_states).squeeze(1).cpu()  # Shape: (num_layers, seq_len, hidden_dim)
            embeddings.append((text_embedding, labels[i]))

            if i % 100 == 0:
                print(f"Processed {i}/{len(texts)}")

    torch.save(embeddings, save_file)

# extract_embeddings(df['text'].tolist(), df['label'].tolist(), os.path.join(save_path, 'train_embeddings.pt'))
# similarly get test embeddings
test_path = '../../HindiSumm/data/test.csv'
test_df = pd.read_csv(test_path)
test_df = test_df.dropna(subset=['text'])
# extract_embeddings(test_df['text'].tolist(), test_df['label'].tolist(), os.path.join(save_path, 'test_embeddings.pt'))
print("Embeddings saved successfully!")

# similarly for test data from xquad
xquad_path = '../../xquad/data/Data_gemini.csv'
xquad_df = pd.read_csv(xquad_path)
xquad_df = xquad_df.dropna(subset=['text'])
# use 200 samples
xquad_df = xquad_df.sample(n=200, random_state=42)
# extract_embeddings(xquad_df['text'].tolist(), xquad_df['label'].tolist(), os.path.join(save_path, 'xquad_embeddings.pt'))
print("XQuAD Embeddings saved successfully!")

# similarly for test data from HindiSumm gpt
gpt_path = '../../HindiSumm/data/Data_gpt.csv'
gpt_df = pd.read_csv(gpt_path)
gpt_df = gpt_df.dropna(subset=['text'])
# extract_embeddings(gpt_df['text'].tolist(), gpt_df['label'].tolist(), os.path.join(save_path, 'gpt_embeddings.pt'))
print("GPT Embeddings saved successfully!")

# llama3_path = '../../HindiSumm/data/test_llama3.csv'
llama3_path = '../../xquad/data/Data_llama3.csv'
llama3_df = pd.read_csv(llama3_path)
llama3_df = llama3_df.dropna(subset=['text'])
extract_embeddings(llama3_df['text'].tolist(), llama3_df['label'].tolist(), os.path.join(save_path, 'llama3_embeddings.pt'))
print("Llama3 Embeddings saved successfully!")







