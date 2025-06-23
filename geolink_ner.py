import pandas as pd
import csv
import evaluate
from sklearn.metrics import classification_report
import torch
from datasets import Dataset, ClassLabel
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification
import torch.nn.functional as F
import requests
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset

# ------------------- Static Configuration -------------------

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
classmap = ClassLabel(num_classes=9, names=label_list)

model_name = "MODEL" # Enter your model here
print(f"📦 Loading model: {model_name}")
model = AutoModelForTokenClassification.from_pretrained(model_name, ignore_mismatched_sizes=True,
                                                        id2label={i: classmap.int2str(i) for i in range(classmap.num_classes)},
                                                        label2id={c: classmap.str2int(c) for c in classmap.names},
                                                        finetuning_task="ner")

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_collator = DataCollatorForTokenClassification(tokenizer)
threshold = 0.00
MAX_WORKERS = min(32, 4 * multiprocessing.cpu_count())

# ------------------- Helper Functions -------------------

def convert_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_tokens, current_labels, current_ids = [], [], []

    for line in lines:
        line = line.strip()
        if line == "":
            if current_tokens:
                sentences.append({
                    "text": " ".join(current_tokens),
                    "labels": current_labels,
                    "ids": current_ids
                })
                current_tokens, current_labels, current_ids = [], [], []
        else:
            parts = line.split()
            if len(parts) == 3:
                token, label, id_ = parts
            elif len(parts) == 2:
                token, label = parts
                id_ = "NA"
            else:
                continue
            label = label if label in label_list else "O"
            current_tokens.append(token)
            current_labels.append(label)
            current_ids.append(id_)

    if current_tokens:
        sentences.append({
            "text": " ".join(current_tokens),
            "labels": current_labels,
            "ids": current_ids
        })

    return sentences

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True, max_length=512)
    all_labels = []
    for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples["text"]))):
        labels, previous_word_idx = [], None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(classmap.str2int(examples["labels"][i][word_idx]))
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        all_labels.append(labels)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def normalize_token(token):
    token = token.lower()
    token = ''.join(c for c in unicodedata.normalize('NFD', token) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w\s]', '', token).strip()

def is_valid_location_qid(qid, language='nl'):
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetentities", "format": "json", "ids": qid, "props": "labels|claims", "languages": language}
    HEADERS = {"User-Agent": "CoordinateRetriever/1.0 (contact@example.com)"}
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return False
    entity = data.get("entities", {}).get(qid, {})
    claims = entity.get("claims", {})
    label = entity.get("labels", {}).get(language, {}).get("value", "")
    if "P625" in claims:
        return (qid, label)
    for prop in ["P131", "P17"]:
        for stmt in claims.get(prop, []):
            fallback_qid = stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if fallback_qid:
                return is_valid_location_qid(fallback_qid, language)
    return False

def get_coordinates(qid):
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetentities", "format": "json", "ids": qid, "props": "claims"}
    HEADERS = {"User-Agent": "CoordinateRetriever/1.0 (contact@example.com)"}
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    entity = data.get("entities", {}).get(qid, {})
    p625 = entity.get("claims", {}).get("P625")
    if not p625:
        return None
    for stmt in p625:
        val = stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if "latitude" in val and "longitude" in val:
            return {"lat": val["latitude"], "lon": val["longitude"]}
    return None

def search_wikidata_qid(entity, cache, language='nl', limit=5):
    if entity in cache:
        return cache[entity]
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "format": "json", "search": entity, "language": language, "limit": limit}
    HEADERS = {"User-Agent": "CoordinateRetriever/1.0 (contact@example.com)"}
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        cache[entity] = {"qid": "O", "label": None, "coords": None}
        return cache[entity]
    for result in data.get("search", []):
        candidate_qid = result["id"]
        resolved = is_valid_location_qid(candidate_qid, language)
        if resolved:
            resolved_qid, label = resolved
            coords = get_coordinates(resolved_qid)
            cache[entity] = {"qid": resolved_qid, "label": label, "coords": coords}
            return cache[entity]
    cache[entity] = {"qid": "O", "label": None, "coords": None}
    return cache[entity]

def parallel_resolve_spans(unique_spans):
    resolved = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(search_wikidata_qid, span, {}): span for span in unique_spans}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🔍 Resolving locations"):
            span = futures[future]
            try:
                resolved[span] = future.result()
            except Exception:
                resolved[span] = {"qid": "O", "coords": None}
    return resolved

def process_tokens_and_predictions(tokens_and_predictions, output_filename):
    lines = [f"{tok[0]}\t{tok[1]}\t{tok[2] if tok[2] else 'NA'}" for tok in tokens_and_predictions]

    spans_with_indices = []
    current_span_tokens, current_span_ids = [], []

    for line in lines:
        if not line.strip():
            if current_span_tokens:
                span_text = " ".join(current_span_tokens)
                spans_with_indices.append((span_text, current_span_ids))
                current_span_tokens, current_span_ids = [], []
            continue

        token, label, id_ = line.split("\t")
        if label == "B-LOC":
            if current_span_tokens:
                span_text = " ".join(current_span_tokens)
                spans_with_indices.append((span_text, current_span_ids))
            current_span_tokens, current_span_ids = [token], [id_]
        elif label == "I-LOC" and current_span_tokens:
            current_span_tokens.append(token)
            current_span_ids.append(id_)
        else:
            if current_span_tokens:
                span_text = " ".join(current_span_tokens)
                spans_with_indices.append((span_text, current_span_ids))
                current_span_tokens, current_span_ids = [], []

    if current_span_tokens:
        span_text = " ".join(current_span_tokens)
        spans_with_indices.append((span_text, current_span_ids))

    unique_spans = {span_text: ids[0] if ids else "NA" for span_text, ids in spans_with_indices}
    resolved = parallel_resolve_spans(unique_spans.keys())

    with open(output_filename, "w", encoding="utf-8") as fout:
        fout.write("Location\tArticleID\tQID\tLabel\tLatitude\tLongitude\n")
        for loc, id_ in unique_spans.items():
            res = resolved.get(loc, {"qid": "O", "coords": None})
            qid, label = res["qid"], res.get("label", "")
            lat = str(res["coords"]["lat"]) if res["coords"] else ""
            lon = str(res["coords"]["lon"]) if res["coords"] else ""
            fout.write(f"{loc}\t{id_}\t{qid}\t{label}\t{lat}\t{lon}\n")

# ------------------- Main File Processing Loop -------------------

for part_num in range(1, 11):
    test_path = f'newscrawl/part_{part_num}_processed.tsv'
    output_filename = f'part_{part_num}_labeled.tsv'
    print(f"\n🚀 Processing {test_path}...")

    eval_sentences = convert_data(test_path)
    ds_eval = Dataset.from_pandas(pd.DataFrame(data=eval_sentences))
    ds_eval = ds_eval.map(lambda x: {"text": x["text"].split()}, batched=False)
    ds_eval = ds_eval.map(tokenize_and_align_labels, batched=True)

    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for i in range(0, len(ds_eval), 16):
            batch = ds_eval[i:i+16]
            inputs = tokenizer(batch["text"], is_split_into_words=True, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_batch = outputs.logits.cpu()
            all_logits.extend([logits_batch[j] for j in range(logits_batch.size(0))])
            all_labels.extend(batch["labels"])

    probs_list = [F.softmax(logit, dim=-1) for logit in all_logits]
    final_preds = []
    for probs in probs_list:
        max_probs, pred_labels = torch.max(probs, dim=-1)
        preds = torch.where(max_probs >= threshold, pred_labels, torch.zeros_like(pred_labels))
        final_preds.append(preds)

    tokens_and_predictions = []
    for sentence_tokens, sentence_predictions in zip(eval_sentences, final_preds):
        tokens = sentence_tokens['text'].split()
        ids = sentence_tokens.get('ids', ['NA'] * len(tokens))
        predicted_labels = [label_list[pred] for pred in sentence_predictions[:len(tokens)]]
        for token, label, id_ in zip(tokens, predicted_labels, ids):
            tokens_and_predictions.append([token, label, id_])

    process_tokens_and_predictions(tokens_and_predictions, output_filename)
    print(f"✅ Saved output: {output_filename}")
