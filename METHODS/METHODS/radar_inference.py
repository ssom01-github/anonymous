import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load tokenizer
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Custom Dataset for Testing
class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512, device="cuda"):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])  # Ensure it's a string
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0).to(self.device),
            'attention_mask': encoding['attention_mask'].squeeze(0).to(self.device),
            'text': text
        }

# Load test data
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop missing values
    return df

# Load trained classifier
class XLMClassifier(torch.nn.Module):
    def __init__(self):
        super(XLMClassifier, self).__init__()
        self.model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

# Load model function
def load_classifier(model_path, device):
    model = XLMClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict function
def test_classifier(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # Convert to NumPy for easier handling
            
            for text, pred in zip(batch['text'], preds):
                predictions.append({'text': text, 'predicted_label': pred})
    
    return pd.DataFrame(predictions)

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = load_classifier("./models/classifier.pth", device)

test_df = load_test_data("../xquad/data/Data_gemini.csv")
test_loader = DataLoader(TestDataset(test_df, xlm_tokenizer, device=device), batch_size=16, shuffle=False)

predictions_df = test_classifier(classifier, test_loader, device)
original_labels = test_df['label'].tolist()
predicted_labels = predictions_df['predicted_label'].tolist()

from sklearn.metrics import classification_report
print(classification_report(original_labels, predicted_labels))

################### PART-2 ###############
print("################### PART-2 ###############")
## UNCOMMENT THIS PART TO CHECK THE PARAPHRASE MODEL WORKING
"""
## CHECKING OF PARAPHRASE MODEL

# import torch
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class Paraphraser(nn.Module):
    def __init__(self):
        super(Paraphraser, self).__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    def forward(self, input_text, tokenizer, max_length=256, device="cuda"):
        if not isinstance(input_text, str):  # Ensure input is a string
            raise ValueError("Expected input_text to be a string, got:", type(input_text))

        # input_text = f"paraphrase in Hindi: {input_text} </s>"
        input_text = f"निम्नलिखित पाठ का अर्थ बनाए रखते हुए, इसकी लंबाई के करीब एक भावार्थ बनाएँ: {input_text}.संक्षिप्त न करें। </s>"
        encoding = tokenizer(
            input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
        ).to(device)

        output_ids = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling
            top_k=50,  # Limit vocabulary for diverse results
            top_p=0.95,  # Nucleus sampling
            temperature=0.7  # Introduce randomness
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

paraphraser = Paraphraser().to(device)
paraphraser.load_state_dict(torch.load("./models/paraphraser.pth", map_location=device))
paraphraser.eval()

# Test with Hindi sentence
# text = "भारत एक महान देश है।"
# paraphrased_text = paraphraser(text, tokenizer, device=device)
# print("Original:", text)
# print("Paraphrased:", paraphrased_text)

test_df = load_test_data("./data/test.csv")
test = test_df['text'].tolist()
test = test[:3]

paraphrased = []
for text in test:
    paraphrased_text = paraphraser(text, tokenizer, device=device)
    paraphrased.append(paraphrased_text)
    
print(test)
print(paraphrased)
"""