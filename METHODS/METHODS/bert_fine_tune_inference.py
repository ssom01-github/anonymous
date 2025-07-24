import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load trained model and tokenizer
model_path = "human_vs_ai_model_gemini"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# Load test data
# test_path = f'../xquad/data/Data_gemini.csv'
test_path = f'./data/test_llama3.csv'
#"../chaii/Data_gemini_100.csv"
test_df = pd.read_csv(test_path)  # Replace with your test CSV file

# Dataset class (same as before)
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