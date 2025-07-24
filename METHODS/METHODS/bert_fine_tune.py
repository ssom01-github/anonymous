import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
Train_data_path = '../../Data/HindiSumm/Data_gemini_2k.csv' # replace with your training data path
Test_data_path = '../../Data/xquad/Data_gemini.csv' # replace with your test data path

# Load data
df = pd.read_csv(Train_data_path)  # Replace with your CSV file
print(f"SIZE:{len(df)}")

# Model and tokenizer
model_name = "xlm-roberta-base"  # Or another suitable Hindi model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Dataset class
class ArticleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare data
texts = df["text"].tolist()
labels = df["label"].tolist()

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets and dataloaders
max_len = 512  # Adjust as needed
train_dataset = ArticleDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = ArticleDataset(val_texts, val_labels, tokenizer, max_len)

batch_size = 16  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3  # Adjust as needed

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
            true = labels.cpu().numpy().tolist()

            val_preds.extend(preds)
            val_true.extend(true)

    accuracy = accuracy_score(val_true, val_preds)
    report = classification_report(val_true, val_preds)

    print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy:.4f}")
    print(report)

# Save the model
model.save_pretrained("human_vs_ai_model_gemini")
tokenizer.save_pretrained("human_vs_ai_model_gemini")



# Load trained model and tokenizer
model_path = "human_vs_ai_model_gemini"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# Load test data
test_df = pd.read_csv(Test_data_path)  # Replace with your test CSV file


# Prepare test data
test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

# Create test dataset and dataloader
max_len = 512  # Should match the training max_len
test_dataset = ArticleDataset(test_texts, test_labels, tokenizer, max_len)
test_dataloader = DataLoader(test_dataset, batch_size=16)  # Adjust batch size if needed

# Evaluation
model.eval()
test_preds = []
test_true = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
        true = labels.cpu().numpy().tolist()

        test_preds.extend(preds)
        test_true.extend(true)

accuracy = accuracy_score(test_true, test_preds)
report = classification_report(test_true, test_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(report)


