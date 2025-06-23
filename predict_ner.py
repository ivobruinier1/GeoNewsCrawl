import pandas as pd
import csv
import evaluate
from sklearn.metrics import classification_report
import torch
from datasets import Dataset
from datasets import ClassLabel
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, EarlyStoppingCallback
import torch.nn.functional as F

# Using the pre-trained model without training
model_name = "model_name_here"
output_directory = model_name.split('/')[-1]
print(f"Using model: {model_name}")

# Step 1: Read the TSV file from the specified path'  # Ensure the path is correct (TSV format)
predict_file = 'file_name_here'


label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def convert_data(file_path, max_tokens=510):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_tokens = []
    current_labels = []

    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    for line in lines:
        line = line.strip()
        if line == "":
            while len(current_tokens) > max_tokens:
                # Slice into manageable chunks
                sentences.append({
                    "text": " ".join(current_tokens[:max_tokens]),
                    "labels": current_labels[:max_tokens]
                })
                current_tokens = current_tokens[max_tokens:]
                current_labels = current_labels[max_tokens:]

            if current_tokens:
                sentences.append({
                    "text": " ".join(current_tokens),
                    "labels": current_labels
                })
                current_tokens = []
                current_labels = []
        else:
            token_label = line.split()
            if len(token_label) == 2:
                token, label = token_label
                if label not in label_list:
                    label = "O"
                current_tokens.append(token)
                current_labels.append(label)

    # Handle any leftover tokens at the end
    while len(current_tokens) > max_tokens:
        sentences.append({
            "text": " ".join(current_tokens[:max_tokens]),
            "labels": current_labels[:max_tokens]
        })
        current_tokens = current_tokens[max_tokens:]
        current_labels = current_labels[max_tokens:]

    if current_tokens:
        sentences.append({
            "text": " ".join(current_tokens),
            "labels": current_labels
        })

    return sentences

eval_sentences = convert_data(predict_file)

# Define a ClassLabel object to use to map string labels to integers.
classmap = ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])

# Convert to Hugging Face Datasets
ds_eval = Dataset.from_pandas(pd.DataFrame(data=eval_sentences))

print(f"Labeling set: {predict_file} \n {ds_eval}")

# Initialize model and tokenizer (using the pre-trained model)
model = AutoModelForTokenClassification.from_pretrained(model_name, ignore_mismatched_sizes=True,
                                                        id2label={i: classmap.int2str(i) for i in range(classmap.num_classes)},
                                                        label2id={c: classmap.str2int(c) for c in classmap.names},
                                                        finetuning_task="ner")

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Tokenize the dataset
ds_eval = ds_eval.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

# Convert labels to integers using ClassLabel mappings
ds_eval = ds_eval.map(lambda y: {"labels": [classmap.str2int(label) for label in y["labels"]]})


# Load the evaluation metric
metric = evaluate.load("seqeval")


def compute_metrics(p):
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    predictions, labels = p
    predictions = torch.tensor(predictions)
    
    # Apply softmax to logits
    probs = F.softmax(predictions, dim=-1)

    # Get max probability and predicted class
    max_probs, pred_classes = torch.max(probs, dim=-1)

    # Apply threshold to filter low-confidence predictions
    threshold = 0.85
    confident_predictions = torch.where(max_probs >= threshold, pred_classes, torch.zeros_like(pred_classes))

    # Convert to string labels for seqeval
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(confident_predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(confident_predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir=output_directory,
    per_device_eval_batch_size=16,  # Required for batching
    do_train=False,  # Explicitly disables training
    do_eval=True,    # Enables evaluation mode
    logging_dir="./logs",  # Optional but useful for logging
    no_cuda=False,
)

# Skip Training, Directly Predicting using the Pre-trained Model
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Get predictions from the pre-trained model
# Get raw logits from the model
logits, labels, _ = trainer.predict(ds_eval)

# Convert logits to torch tensor
logits = torch.tensor(logits).float()

# Apply softmax to get probabilities
probs = F.softmax(logits, dim=-1)

# Get predicted classes and their max probability
max_probs, pred_labels = torch.max(probs, dim=-1)

# Apply threshold
threshold = 0.85
final_preds = torch.where(max_probs >= threshold, pred_labels, torch.zeros_like(pred_labels))  # Use 'O' label index (0) if below threshold

# Write predictions to file
tokens_and_predictions = []

for sentence_tokens, sentence_predictions in zip(eval_sentences, final_preds):
    tokens = sentence_tokens['text'].split()
    predicted_labels = [label_list[pred.item()] for pred in sentence_predictions[:len(tokens)]]

    for token, label in zip(tokens, predicted_labels):
        tokens_and_predictions.append([token, label])
    tokens_and_predictions.append([])  # Blank line between sentences

# Write the results to a CSV file
with open(f'{output_directory}/cc_val_labeled.tsv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(tokens_and_predictions)

print(f"Predictions written to {output_directory}/.tsv")
