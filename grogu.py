import pandas as pd
import evaluate
from sklearn.metrics import classification_report
import torch
from datasets import Dataset
from datasets import ClassLabel
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, EarlyStoppingCallback

model_name = "checkpoints/checkpoint-310000"
output_directory = "pretrained-robbert-v2"
print(f"Using model: {model_name}")

# Step 1: Read the CSV file from the specified path
train_path = 'tsv_data/EWN/EWN_train.tsv'  # Make sure the path is correct
test_path = 'tsv_data/cc_news_test.tsv'

def convert_data(file_path):
    # Read the TSV file with space-separated tokens and labels
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 2: Initialize lists to hold the transformed sentences
    sentences = []
    current_tokens = []
    current_labels = []

    end_token = ["?", "!", "."]
    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC','B-MISC','I-MISC']

    # Step 3: Process the TSV file line by line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line == "":  # If it's a blank line, consider it the end of a sentence
            if current_tokens:
                sentences.append({"text": " ".join(current_tokens), "labels": current_labels})
                current_tokens = []
                current_labels = []
        else:
            token_label = line.split()  # Split line by whitespace
            if len(token_label) == 2:
                token, label = token_label
                if label not in label_list:
                    current_tokens.append(token)
                    current_labels.append("O")
                else:
                    current_tokens.append(token)
                    current_labels.append(label)

    # Handle the case if there's no punctuation at the end of a sentence
    if current_tokens:
        sentences.append({"text": " ".join(current_tokens), "labels": current_labels})

    return sentences


train_sentences = convert_data(train_path)
eval_sentences = convert_data(test_path)

# Define a ClassLabel object to use to map string labels to integers.
classmap = ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])

# Create the label_list variable
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


# Convert to Hugging Face Datasets
ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
ds_eval = Dataset.from_pandas(pd.DataFrame(data=eval_sentences))

print(ds_train)

# Initialize model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(model_name, ignore_mismatched_sizes=True,
                                                        id2label={i: classmap.int2str(i) for i in
                                                                  range(classmap.num_classes)},
                                                        label2id={c: classmap.str2int(c) for c in classmap.names},
                                                        finetuning_task="ner")

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Tokenize the dataset
ds_train = ds_train.map(lambda x: tokenizer(x["text"], truncation=True))
ds_eval = ds_eval.map(lambda x: tokenizer(x["text"], truncation=True))

# Convert labels to integers using ClassLabel mappings
ds_train = ds_train.map(lambda y: {"labels": [classmap.str2int(label) for label in y["labels"]]})
ds_eval = ds_eval.map(lambda y: {"labels": [classmap.str2int(label) for label in y["labels"]]})

# Load the evaluation metric
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir=output_directory,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    no_cuda=False

)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

# Train the model
trainer.train()

# Get predictions from the model
predictions, labels, _ = trainer.predict(ds_eval)

# Process the predictions
predictions = predictions.argmax(axis=2)

# Remove special tokens and prepare data for classification report
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# Flatten the lists of labels and predictions to pass to classification_report
flat_true_predictions = [item for sublist in true_predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]

label_order = [
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    "B-MISC", "I-MISC",
    "O"
]

# Print classification report
report = classification_report(flat_true_labels, flat_true_predictions, labels=label_order)
with open(f'{output_directory}/pretraining_on_ccnewstest.txt', 'w') as f:
    f.write(f' Results and settings for {output_directory}:')
    f.write("\n\n" + report + "\n\n")
    f.write("\n".join(f"{k}: {v}" for k, v in vars(training_args).items()))

print(report)


