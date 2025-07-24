# -*- coding: utf-8 -*-
import numpy as np
import re
import transformers
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import functools
import time
from indicnlp.tokenize import indic_tokenize  # For Hindi tokenization

# Device setup
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model and tokenizer settings
mask_filling_model_name = 'google/mt5-base'  # Multilingual model supporting Hindi
base_model_name = 'xlm-roberta-base'  # Multilingual model supporting Hindi
cache_dir = None
batch_size = 200
n_samples = 4000
n_perturbation_list = [1, 10]  # List of perturbation counts
span_length = 2
pct_words_masked = 0.25
chunk_size = 20
buffer_size = 1
mask_top_p = 1.0
pattern = re.compile(r"<extra_id_\d+>")
n_perturbation_rounds = 1

def strip_newlines(text):
    return ' '.join(text.split())

def get_ll(text):
    if not text or text.isspace():  # Skip empty or whitespace-only text
        return None
    encodings = base_tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    lls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, 512):
        end_loc = min(begin_loc + 512, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = base_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        lls.append(-neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    result = torch.stack(lls).sum() / end_loc
    return result.item()

def get_lls(texts):
    return [get_ll(text) for text in texts if text and not text.isspace()]

def get_rank(text, log=False):
    if not text or text.isspace():
        return None
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        ranks, timesteps = matches[:, -1], matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float() + 1
        if log:
            ranks = torch.log(ranks)
        return ranks.float().mean().item()

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    if not text or text.isspace():
        return text
    # Use Indic NLP for Hindi tokenization
    tokens = indic_tokenize.trivial_tokenize(text, lang='hi')
    mask_string = '<<<mask>>>'
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    tokens = [x.split(' ') for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    texts = [" ".join(x) for x in tokens]
    return texts

def reduce_pct(pct, attempt):
    return pct - 0.05 * attempt

def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts if x and not x.isspace()]
    if not masked_texts:
        return []
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, reduce_pct(pct, attempts), ceil_pct) for idx, x in enumerate(texts) if idx in idxs and x and not x.isspace()]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def perturb_texts(texts, span_length, pct, ceil_pct=False):
    outputs = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Perturbating"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs

def n_get_perturbation_results(dataset_path, output_path, span_length=10, n_perturbations=1):
    load_mask_model()
    torch.manual_seed(0)
    np.random.seed(0)
    data = pd.read_csv(dataset_path)
    # drop rows with empty text
    data = data.dropna(subset=['text'])
    # Filter out empty or whitespace-only text
    data = data[data['text'].str.strip().astype(bool)]
    text = data['text'].tolist()
    text = [strip_newlines(x.strip()) for x in text]
    data['text'] = data['text'].str.strip().apply(strip_newlines)
    
    tokenized_text = preproc_tokenizer(text, max_length=512, truncation=True).input_ids
    text = preproc_tokenizer.batch_decode(tokenized_text, skip_special_tokens=True)
    
    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=pct_words_masked)
    p_text = perturb_fn([x for x in text for _ in range(n_perturbations)])
    
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_text = perturb_fn(p_text)
        except AssertionError:
            break
    
    assert len(p_text) == len(text) * n_perturbations, f"Expected {len(text) * n_perturbations} perturbed samples, got {len(p_text)}"
    
    results = []
    for idx in range(len(text)):
        results.append({
            "original_generation": data.iloc[idx]['text'],
            "generation": text[idx],
            "perturbed_generation": p_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })
    
    load_base_model()
    for res in tqdm(results, desc="Computing log likelihoods"):
        loc = data[data['text'] == res["original_generation"]].index[0]
        p_ll = get_lls(res["perturbed_generation"])
        res['p_ll_list'] = p_ll
        res["ll"] = get_ll(res["generation"])
        res["perturbed_ll"] = np.mean([x for x in p_ll if x is not None]) if any(x is not None for x in p_ll) else None
        res['std'] = np.std([x for x in p_ll if x is not None]) if any(x is not None for x in p_ll) else 0
        if res['std'] != 0:
            data.at[loc, 'llr'] = (res["ll"] - res["perturbed_ll"]) / res['std'] if res["ll"] is not None and res["perturbed_ll"] is not None else None
        else:
            data.at[loc, 'llr'] = (res["ll"] - res["perturbed_ll"]) if res["ll"] is not None and res["perturbed_ll"] is not None else None
    data.to_csv(output_path, index=False)
    return results

def load_base_model():
    base_model.to(DEVICE)

def load_mask_model():
    mask_model.to(DEVICE)

def load_base_model_and_tokenizer(name):
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer

# Load models and tokenizers
base_model, base_tokenizer = load_base_model_and_tokenizer('xlm-roberta-base')
mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('google/mt5-base', cache_dir=cache_dir)
mask_tokenizer = transformers.AutoTokenizer.from_pretrained('google/mt5-base', cache_dir=cache_dir, model_max_length=512)
preproc_tokenizer = mask_tokenizer
load_base_model()

# Process Hindi text
# source_dir = '../HindiSumm/data'
# target_dir = '../HindiSumm/data_with_detectgpt_scores'

source_dir = '../xquad/data/'
target_dir = '../xquad/data_with_detectgpt_scores'

os.makedirs(target_dir, exist_ok=True)

input_file = f"{source_dir}/Data_llama3.csv"
output_file = f"{target_dir}/Data_llama3_detectgpt.csv"
perturbation_results = n_get_perturbation_results(input_file, output_file, span_length, n_perturbations=60)

pkl_file_path = f'{target_dir}/output'
os.makedirs(pkl_file_path, exist_ok=True)
with open(f'{pkl_file_path}/train_output.pkl', 'wb') as f:
    pickle.dump(perturbation_results, f)