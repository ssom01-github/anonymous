import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from functools import reduce
from tqdm import tqdm
import pandas as pd
import numpy as np
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import re
import unicodedata
import logging
from collections import namedtuple
import argparse
import subprocess
from itertools import count, cycle
from multiprocessing import Process

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create output directory
output_dir = 'conda_hindi_xquad'
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to both terminal and output.txt
output_file_path = os.path.join(output_dir, 'output_conda.txt')
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

# MMD implementation (unchanged)
def MMD(x, y, kernel):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)

# Preprocessing for Hindi
class PreProcess:
    def __init__(self, lowercase_norm=False, special_chars_norm=True, accented_norm=True, stopword_norm=False):
        self.lowercase_norm = lowercase_norm
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.stopword_norm = stopword_norm
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        # Hindi stopwords (sample list, expand as needed)
        self.stopwords = set([
            'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होगा', 'होता', 'होती', 'के', 'का', 'की', 'को', 'में', 'से', 'पर', 'और', 'या', 'लेकिन'
        ])

    def normalize_hindi(self, text):
        # Normalize Hindi text (e.g., handle Devanagari script variations)
        return self.normalizer.normalize(text)

    def tokenize_hindi(self, text):
        # Tokenize Hindi text using indic-nlp
        return indic_tokenize.trivial_tokenize(text, lang='hi')

    def special_char_remove(self, text):
        # Remove special characters, keep Hindi characters
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep Devanagari and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        return text

    def accented_word_normalization(self, text):
        # Normalize accented characters (e.g., combining diacritics)
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
            text = text.lower()  # Optional for Hindi, as Devanagari doesn't use case
        
        return text

# Data loading for CSV
def load_texts(data_file, label=True):
    df = pd.read_csv(data_file)
    if not all(col in df.columns for col in ['text', 'text_perturb']):
        raise ValueError(f"CSV file {data_file} must contain 'text' and 'text_perturb' columns")
    
    texts = df['text'].tolist()
    texts_perturb = df['text_perturb'].tolist()
    
    if label and 'label' not in df.columns:
        raise ValueError(f"CSV file {data_file} must contain 'label' column for labeled data")
    
    labels = df['label'].tolist() if label else None
    return (texts, texts_perturb, labels) if label else (texts, texts_perturb)

class Corpus:
    def __init__(self, name, data_dir, label=True, single_file=False):
        self.name = name
        if single_file:
            if label:
                self.data, self.data_perturb, self.label = load_texts(f'{data_dir}/{name}_perturb.csv', label=True)
            else:
                self.data, self.data_perturb = load_texts(f'{data_dir}/{name}_perturb.csv')
        else:
            self.train, self.train_perturb = load_texts(f'{data_dir}/{name}_train_perturb.csv')
            self.test, self.test_perturb = load_texts(f'{data_dir}/{name}_test_perturb.csv')
            self.valid, self.valid_perturb = load_texts(f'{data_dir}/{name}_valid_perturb.csv')

class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], real_texts_perturb: List[str],
                 fake_texts: List[str], fake_texts_perturb: List[str],
                 tokenizer, max_sequence_length: int = None, min_sequence_length: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.real_text_perturb = real_texts_perturb
        self.fake_text_perturb = fake_texts_perturb
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            text_perturb = self.real_text_perturb[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            text_perturb = self.fake_text_perturb[index - len(self.real_texts)]
            label = 0

        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        text_perturb = preprocessor.fit(text_perturb)

        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        padded_sequences_perturb = self.tokenizer(text_perturb, padding='max_length', max_length=self.max_sequence_length, truncation=True)

        return (torch.tensor(padded_sequences['input_ids']),
                torch.tensor(padded_sequences['attention_mask']),
                torch.tensor(padded_sequences_perturb['input_ids']),
                torch.tensor(padded_sequences_perturb['attention_mask']),
                label)

class EncodedSingleDataset(Dataset):
    def __init__(self, input_texts: List[str], input_labels: List[int], tokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None):
        self.input_texts = input_texts
        self.input_labels = input_labels
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

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

class EncodeEvalData(Dataset):
    def __init__(self, input_texts: List[str], tokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        text = self.input_texts[index]
        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        return (torch.tensor(padded_sequences['input_ids']),
                torch.tensor(padded_sequences['attention_mask']))

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

# Projection MLP
class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 300))

    def forward(self, input_features):
        x = input_features[:, 0, :]
        return self.layers(x)

# SimCLR Contrastive Loss
class SimCLRContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        if denominator.sum(dim=1).eq(0).any():
            raise ValueError("Denominator contains zero, causing division by zero")
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return torch.sum(loss_partial) / (2 * self.batch_size)

# Contrastive Model
class ContrastivelyInstructedXLMRoberta(nn.Module):
    def __init__(self, model: nn.Module, mlp: nn.Module, loss_type: str, logger: SummaryWriter, device: str, lambda_w: float):
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.loss_type = loss_type
        self.logger = logger
        self.device = device
        self.lambda_w = lambda_w

    def forward(self, src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                src_labels, tgt_labels):
        batch_size = src_texts.shape[0]
        src_output_dic = self.model(src_texts, attention_mask=src_masks, labels=src_labels)
        src_LCE_real, src_logits_real = src_output_dic["loss"], src_output_dic["logits"]
        src_output_dic_perturbed = self.model(src_texts_perturb, attention_mask=src_masks_perturb, labels=src_labels)
        src_LCE_perturb, src_logits_perturb = src_output_dic_perturbed["loss"], src_output_dic_perturbed["logits"]
        tgt_output_dic = self.model(tgt_texts, attention_mask=tgt_masks, labels=tgt_labels)
        tgt_LCE_real, tgt_logits_real = tgt_output_dic["loss"], tgt_output_dic["logits"]
        tgt_output_dic_perturbed = self.model(tgt_texts_perturb, attention_mask=tgt_masks_perturb, labels=tgt_labels)
        tgt_LCE_perturb, tgt_logits_perturb = tgt_output_dic_perturbed["loss"], tgt_output_dic_perturbed["logits"]
        
        if self.loss_type == "simclr":
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size).to(self.device)
            src_z_i = self.mlp(src_output_dic["last_hidden_state"])
            src_z_j = self.mlp(src_output_dic_perturbed["last_hidden_state"])
            src_lctr = ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_output_dic["last_hidden_state"])
            tgt_z_j = self.mlp(tgt_output_dic_perturbed["last_hidden_state"])
            tgt_lctr = ctr_loss(tgt_z_i, tgt_z_j)
        
        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf')
        use_ce_perturb = True
        use_both_ce_losses = True
        lambda_mmd = 1.0
        
        if not use_both_ce_losses:
            loss = self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        else:
            if use_ce_perturb:
                loss = (1 - self.lambda_w) * (src_LCE_real + src_LCE_perturb) / 2 + \
                       self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
            else:
                loss = (1 - self.lambda_w) * src_LCE_real + self.lambda_w * (src_lctr + tgt_lctr) / 2 + lambda_mmd * mmd
        
        data = {
            "total_loss": loss,
            "src_ctr_loss": src_lctr,
            "tgt_ctr_loss": tgt_lctr,
            "src_ce_loss_real": src_LCE_real,
            "src_ce_loss_perturb": src_LCE_perturb,
            "mmd": mmd,
            "src_logits": src_logits_real,
            "tgt_logits": tgt_logits_real
        }
        data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
        return data_named_tuple(**data)

# Utility functions
def summary(model: nn.Module, file=sys.stdout):
    def repr(model):
        extra_lines = model.extra_repr().split('\n') if model.extra_repr() else []
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)
        lines = extra_lines + child_lines
        main_str = model._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        main_str += f', {total_params:,} params' if file is sys.stdout else f', {total_params:,} params'
        return main_str, total_params
    
    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()
    return count

def grad_norm(model: nn.Module):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def distributed():
    return False  # Disable distributed training for simplicity

# Data loading for training
def load_datasets(src_data_dir, tgt_data_dir, tokenizer, batch_size, max_sequence_length, random_sequence_length):
    # Source data
    src_df = pd.read_csv(os.path.join(src_data_dir, 'train_perturb.csv'))
    src_real = src_df[src_df['label'] == 1]
    src_fake = src_df[src_df['label'] == 0]
    src_real_texts, src_real_perturb = src_real['text'].tolist(), src_real['text_perturb'].tolist()
    src_fake_texts, src_fake_perturb = src_fake['text'].tolist(), src_fake['text_perturb'].tolist()
    
    # Target data
    tgt_df = pd.read_csv(os.path.join(tgt_data_dir, 'Data_gemini_perturb.csv'))
    tgt_real = tgt_df[tgt_df['label'] == 1]
    tgt_fake = tgt_df[tgt_df['label'] == 0]
    tgt_real_texts, tgt_real_perturb = tgt_real['text'].tolist(), tgt_real['text_perturb'].tolist()
    tgt_fake_texts, tgt_fake_perturb = tgt_fake['text'].tolist(), tgt_fake['text_perturb'].tolist()
    
    # Split source data into train/valid (80-20 split)
    from sklearn.model_selection import train_test_split
    src_real_train, src_real_valid, src_real_perturb_train, src_real_perturb_valid = train_test_split(
        src_real_texts, src_real_perturb, test_size=0.2, random_state=42)
    src_fake_train, src_fake_valid, src_fake_perturb_train, src_fake_perturb_valid = train_test_split(
        src_fake_texts, src_fake_perturb, test_size=0.2, random_state=42)
    
    # Use all target data for training (no validation split for simplicity)
    tgt_real_train, tgt_real_perturb_train = tgt_real_texts, tgt_real_perturb
    tgt_fake_train, tgt_fake_perturb_train = tgt_fake_texts, tgt_fake_perturb
    
    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler
    min_sequence_length = 10 if random_sequence_length else None
    
    src_train_dataset = EncodedDataset(src_real_train, src_real_perturb_train, src_fake_train, src_fake_perturb_train,
                                      tokenizer, max_sequence_length, min_sequence_length)
    src_train_loader = DataLoader(src_train_dataset, batch_size, sampler=Sampler(src_train_dataset), num_workers=0, drop_last=True)
    
    src_valid_dataset = EncodedDataset(src_real_valid, src_real_perturb_valid, src_fake_valid, src_fake_perturb_valid,
                                      tokenizer, max_sequence_length, min_sequence_length)
    src_valid_loader = DataLoader(src_valid_dataset, batch_size=1, sampler=Sampler(src_valid_dataset))
    
    tgt_train_dataset = EncodedDataset(tgt_real_train, tgt_real_perturb_train, tgt_fake_train, tgt_fake_perturb_train,
                                      tokenizer, max_sequence_length, min_sequence_length)
    tgt_train_loader = DataLoader(tgt_train_dataset, batch_size, sampler=Sampler(tgt_train_dataset), num_workers=0, drop_last=True)
    
    return src_train_loader, src_valid_loader, tgt_train_loader

# Training function
def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()

def train(model, mlp, loss_type, optimizer, device, src_loader, tgt_loader, summary_writer, desc='Train', lambda_w=0.5):
    model.train()
    src_train_accuracy = 0
    tgt_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0
    
    if len(src_loader) == len(tgt_loader):
        double_loader = enumerate(zip(src_loader, tgt_loader))
    elif len(src_loader) < len(tgt_loader):
        double_loader = enumerate(zip(cycle(src_loader), tgt_loader))
    else:
        double_loader = enumerate(zip(src_loader, cycle(tgt_loader)))
    
    with tqdm(double_loader, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop:
        torch.cuda.empty_cache()
        for i, (src_data, tgt_data) in loop:
            src_texts, src_masks, src_texts_perturb, src_masks_perturb, src_labels = src_data
            src_texts, src_masks, src_labels = src_texts.to(device), src_masks.to(device), src_labels.to(device)
            src_texts_perturb, src_masks_perturb = src_texts_perturb.to(device), src_masks_perturb.to(device)
            batch_size = src_texts.shape[0]
            
            tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb, tgt_labels = tgt_data
            tgt_texts, tgt_masks, tgt_labels = tgt_texts.to(device), tgt_masks.to(device), tgt_labels.to(device)
            tgt_texts_perturb, tgt_masks_perturb = tgt_texts_perturb.to(device), tgt_masks_perturb.to(device)
            
            optimizer.zero_grad()
            output_dic = model(src_texts, src_masks, src_texts_perturb, src_masks_perturb,
                              tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,
                              src_labels, tgt_labels)
            loss = output_dic.total_loss
            loss.backward()
            optimizer.step()
            
            src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
            src_train_accuracy += src_batch_accuracy
            tgt_batch_accuracy = accuracy_sum(output_dic.tgt_logits, tgt_labels)
            tgt_train_accuracy += tgt_batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size
            
            if i % 25 == 0:
                postfix_str = f"Iteration {i}/{len(src_loader)}: {desc}: loss={loss.item()}, src_acc={src_train_accuracy / train_epoch_size}, tgt_acc={tgt_train_accuracy / train_epoch_size}, mmd={output_dic.mmd.item()}, src_LCE_real={output_dic.src_ce_loss_real.item()}, src_LCE_perturb={output_dic.src_ce_loss_perturb.item()}"
                print(postfix_str)
                with open(os.path.join(output_dir, 'conda_print.txt'), 'a') as f:
                    f.write(postfix_str + '\n')
    
    return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }

def validate(model, device, loader, votes=1, desc='Validation'):
    model.eval()
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    
    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                               disable=distributed() and dist.get_rank() > 0)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]
    
    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []
            for texts, masks, texts_perturb, masks_perturb, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]
                output_dic = model(texts, attention_mask=masks, labels=labels)
                loss, logits = output_dic["loss"], output_dic["logits"]
                losses.append(loss)
                logit_votes.append(logits)
            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)
            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size
            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)
    
    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }

def _all_reduce_dict(d, device):
    output_d = {}
    for key, value in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        output_d[key] = tensor_input.item()
    return output_d

# Main training function
def run(src_data_dir, tgt_data_dir, model_save_path, model_save_name, batch_size, loss_type,
        max_epochs=None, device=None, max_sequence_length=512, random_sequence_length=False,
        learning_rate=2e-5, weight_decay=0, load_from_checkpoint=True, lambda_w=0.5,
        checkpoint_name='checkpoint'):
    args = locals()
    rank, world_size = 0, 1  # No distributed training
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'rank: {rank}, world_size: {world_size}, device: {device}')
    
    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir) if rank == 0 else None
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    xlm_model = XLMRobertaForContrastiveClassification.from_pretrained('xlm-roberta-base').to(device)
    mlp = ProjectionMLP().to(device)
    model = ContrastivelyInstructedXLMRoberta(model=xlm_model, mlp=mlp, loss_type=loss_type,
                                              logger=writer, device=device, lambda_w=lambda_w)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start_epoch = 0
    best_validation_accuracy = 0
    if load_from_checkpoint:
        checkpoint_path = os.path.join(model_save_path, checkpoint_name)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            xlm_model.load_state_dict(checkpoint['model_state_dict'])
            mlp.load_state_dict(checkpoint['mlp_state_dict'])
            model = ContrastivelyInstructedXLMRoberta(model=xlm_model, mlp=mlp, loss_type=loss_type,
                                                      logger=writer, device=device, lambda_w=lambda_w)
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            combined_metrics = checkpoint['metrics_state_dict']
            best_validation_accuracy = combined_metrics["validation/accuracy"]
            print(f">>>>>>> Resuming training from checkpoint epoch {start_epoch} <<<<<<<<<<<<<")
    
    if rank == 0:
        summary(model)
    
    src_train_loader, src_valid_loader, tgt_train_loader = load_datasets(
        src_data_dir, tgt_data_dir, tokenizer, batch_size, max_sequence_length, random_sequence_length)
    
    epoch_loop = range(start_epoch, 4) if max_epochs is None else range(start_epoch, max_epochs + 1)
    without_progress = 0
    earlystop_epochs = 10
    
    for epoch in epoch_loop:
        train_metrics = train(model, mlp, loss_type, optimizer, device, src_train_loader, tgt_train_loader,
                              writer, f'Epoch {epoch}', lambda_w=lambda_w)
        validation_metrics = validate(xlm_model, device, src_valid_loader)
        
        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)
        combined_metrics["train/src_accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]
        
        if rank == 0:
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)
            
            model_to_save = xlm_model.module if hasattr(xlm_model, 'module') else xlm_model
            mlp_to_save = mlp.module if hasattr(mlp, 'module') else mlp
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'mlp_state_dict': mlp_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics_state_dict': combined_metrics,
                'args': args
            }, os.path.join(model_save_path, checkpoint_name))
            print(f"Checkpoint saved for epoch {epoch}")
            
            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                without_progress = 0
                best_validation_accuracy = combined_metrics["validation/accuracy"]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'mlp_state_dict': mlp_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics_state_dict': combined_metrics,
                    'args': args
                }, os.path.join(model_save_path, model_save_name))
                print(f"Best Model Saved for epoch {epoch}")
        
        without_progress += 1
        print(f"without progress- {without_progress}")
        
        if without_progress >= earlystop_epochs:
            break
    
    print(">>>>>>>>>>>>> Training Completed <<<<<<<<<<<<<<")
    if writer:
        writer.close()
    output_file.close()

if __name__ == "__main__":
    src_data_dir = '../HindiSumm/data'
    tgt_data_dir = '../xquad/data'
    model_save_path = 'conda_hindi_xquad'
    model_save_name = 'conda_hindi_xquad.pt'
    checkpoint_name = 'checkpoint_conda_hindi_xquad.pt'
    batch_size = 8
    loss_type = 'simclr'
    max_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 2e-5
    lambda_w = 0.5
    
    run(
        src_data_dir=src_data_dir,
        tgt_data_dir=tgt_data_dir,
        model_save_path=model_save_path,
        model_save_name=model_save_name,
        batch_size=batch_size,
        loss_type=loss_type,
        max_epochs=max_epochs,
        device=device,
        max_sequence_length=512,
        random_sequence_length=False,
        learning_rate=learning_rate,
        weight_decay=0,
        load_from_checkpoint=False,
        lambda_w=lambda_w,
        checkpoint_name=checkpoint_name
    )