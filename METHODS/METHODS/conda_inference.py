import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List
from tqdm import tqdm
import pandas as pd
import numpy as np
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re
import unicodedata
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Output directory
output_dir = '.results/conda_hindi_gemini_on_xquad'
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to both terminal and test_results.txt
output_file_path = os.path.join(output_dir, 'test_results.txt')
output_file = open(output_file_path, 'w')

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, output_file)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for Hindi
class PreProcess:
    def __init__(self, lowercase_norm=False, special_chars_norm=True, accented_norm=True, stopword_norm=False):
        self.lowercase_norm = lowercase_norm
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.stopword_norm = stopword_norm
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        self.stopwords = set([
            'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होगा', 'होता', 'होती', 'के', 'का', 'की', 'को', 'में', 'से', 'पर', 'और', 'या', 'लेकिन'
        ])

    def normalize_hindi(self, text):
        return self.normalizer.normalize(text)

    def tokenize_hindi(self, text):
        return indic_tokenize.trivial_tokenize(text, lang='hi')

    def special_char_remove(self, text):
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def accented_word_normalization(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def stopword_remove(self, text):
        tokens = self.tokenize_hindi(text)
        return ' '.join(word for word in tokens if word not in self.stopwords)

    def fit(self, text):
        text = str(text)
        text = self.normalize_hindi(text)
        
        if self.special_chars_norm:
            text = self.special_char_remove(text)
        
        if self.accented_norm:
            text = self.accented_word_normalization(text)
        
        if self.stopword_norm:
            text = self.stopword_remove(text)
        
        if self.lowercase_norm:
            text = text.lower()
        
        return text

# Data loading for CSV
def load_test_texts(data_file):
    df = pd.read_csv(data_file)
    if not all(col in df.columns for col in ['text', 'text_perturb', 'label']):
        raise ValueError(f"CSV file {data_file} must contain 'text', 'text_perturb', and 'label' columns")
    
    texts = df['text'].tolist()
    texts_perturb = df['text_perturb'].tolist()
    labels = df['label'].tolist()
    return texts, texts_perturb, labels

# Dataset for evaluation
class EncodeEvalData(Dataset):
    def __init__(self, input_texts: List[str], input_labels: List[int], tokenizer,
                 max_sequence_length: int = None):
        self.input_texts = input_texts
        self.input_labels = input_labels
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        text = self.input_texts[index]
        label = self.input_labels[index]
        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        return (torch.tensor(padded_sequences['input_ids']),
                torch.tensor(padded_sequences['attention_mask']),
                label)

# Model definition
@dataclass
class SequenceClassifierOutputWithLastLayer(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class XLMRobertaForContrastiveClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        softmax_logits = self.soft_max(logits)
        if not return_dict:
            output = (softmax_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithLastLayer(
            loss=loss,
            logits=softmax_logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Evaluation function
def evaluate(model, device, loader, desc='Test'):
    model.eval()
    test_accuracy = 0
    test_loss = 0
    test_epoch_size = 0
    all_preds = []
    all_labels = []

    with tqdm(loader, desc=desc) as loop, torch.no_grad():
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]
            output_dic = model(texts, attention_mask=masks, labels=labels)
            loss, logits = output_dic["loss"], output_dic["logits"]
            
            # Compute predictions
            if list(logits.shape) == list(labels.shape) + [2]:
                preds = (logits[..., 0] < logits[..., 1]).long().flatten()
            else:
                preds = (logits > 0).long().flatten()
            
            batch_accuracy = (preds == labels).float().sum().item()
            test_accuracy += batch_accuracy
            test_loss += loss.item() * batch_size
            test_epoch_size += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item(), acc=test_accuracy / test_epoch_size)

    test_accuracy /= test_epoch_size
    test_loss /= test_epoch_size
    
    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'])
    
    return {
        'test/accuracy': test_accuracy,
        'test/loss': test_loss,
        'test/epoch_size': test_epoch_size,
        'confusion_matrix': cm,
        'classification_report': report
    }

# Main test function
def run_test(test_data_file, model_save_path, model_save_name, batch_size, max_sequence_length=512, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on device: {device}")
    
    # Load model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    xlm_model = XLMRobertaForContrastiveClassification.from_pretrained('xlm-roberta-base').to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_save_path, model_save_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    xlm_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    
    # Load test data
    texts, texts_perturb, labels = load_test_texts(test_data_file)
    test_dataset = EncodeEvalData(texts, labels, tokenizer, max_sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    metrics = evaluate(xlm_model, device, test_loader, desc='Testing')
    
    # Print and save results
    print("\nTest Results:")
    print(f"Accuracy: {metrics['test/accuracy']:.4f}")
    print(f"Loss: {metrics['test/loss']:.4f}")
    print(f"Total samples: {metrics['test/epoch_size']}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Save results to file
    with open(output_file_path, 'a') as f:
        f.write("\nTest Results:\n")
        f.write(f"Accuracy: {metrics['test/accuracy']:.4f}\n")
        f.write(f"Loss: {metrics['test/loss']:.4f}\n")
        f.write(f"Total samples: {metrics['test/epoch_size']}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + '\n')
        f.write("\nClassification Report:\n")
        f.write(metrics['classification_report'] + '\n')
    
    output_file.close()

if __name__ == "__main__":
    test_data_file = '../HindiSumm/data/test_perturb.csv'
    model_save_path = './models/conda_hindi_xquad'
    model_save_name = 'conda_hindi_xquad.pt'
    batch_size = 8
    max_sequence_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_test(
        test_data_file=test_data_file,
        model_save_path=model_save_path,
        model_save_name=model_save_name,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        device=device
    )