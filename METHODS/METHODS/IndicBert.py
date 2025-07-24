import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
import joblib

# Paths
train_path = '../HindiSumm/data/train.csv'
val_path = '../HindiSumm/data/val.csv'
test_path = '../HindiSumm/data/test.csv'
model_save_path = './models/hindi_indicbert_model.pkl'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IndicBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=2).to(device)

# Data loading
train_data = pd.read_csv(train_path).dropna()
val_data = pd.read_csv(val_path).dropna()
test_data = pd.read_csv(test_path).dropna()

# Preprocessing: Remove zero-width spaces and convert Hindi digits to English
def preprocess_text(text):
    text = text.replace('\u200b', '')
    hindi_to_eng_digits = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(hindi_to_eng_digits)

train_data['text'] = train_data['text'].apply(preprocess_text)
val_data['text'] = val_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Custom Dataset Class
class HindiDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Prepare datasets
train_dataset = HindiDataset(train_data['text'].tolist(), train_data['label'].values)
val_dataset = HindiDataset(val_data['text'].tolist(), val_data['label'].values)
test_dataset = HindiDataset(test_data['text'].tolist(), test_data['label'].values)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop with Early Stopping
best_val_acc = 0
patience = 2
stopping_counter = 0

## COMEMNTED FOR TESTING PURPOSES, 
## UNCOMMENT FOR TRAINING
"""
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Early Stopping Logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        joblib.dump(model, model_save_path)
        print("Model saved.")
        stopping_counter = 0
    else:
        stopping_counter += 1

    if stopping_counter >= patience:
        print("Early stopping triggered.")
        break

print("Training complete.")

# Load best model for testing
model = joblib.load(model_save_path).to(device)
model.eval()

# Evaluate on Test Set
all_preds, all_labels = [], test_data['label'].values

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        all_preds.extend(preds.cpu().numpy())

print("\nTest Set Evaluation:")
print(classification_report(all_labels, all_preds))

"""

# load the model
model = joblib.load(model_save_path)

# new_test_data_path = f'../xquad/data/Data_gemini.csv'
# new_test_data_path = f'../xquad/data/Data_llama3.csv'
# new_test_data_path = f'../HindiSumm/data/test_llama3.csv'
new_test_data_path = f'../HindiSumm/data/test.csv'


# new_test_data = pd.read_csv(new_test_data_path).dropna()
new_test_data = pd.read_csv(new_test_data_path)
new_test_data['text'] = new_test_data['text'].apply(preprocess_text)

new_test_dataset = HindiDataset(new_test_data['text'].tolist(), new_test_data['label'].values)
new_test_loader = DataLoader(new_test_dataset, batch_size=16, shuffle=False)

all_preds, all_labels = [], new_test_data['label'].values

with torch.no_grad():
    for batch in new_test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        all_preds.extend(preds.cpu().numpy())

print("\nTest Set Evaluation:")
print(classification_report(all_labels, all_preds))

