import os
import pandas as pd
import json
from urllib.parse import urlparse

# Directory containing JSON files
input_dir = "common_crawl_news"  # Change this to your actual folder path
output_filename = "common_crawl_news.json"

# List to store all sampled articles
all_samples = []

# Function to extract the first 5 sentences
def get_first_sentences(text):
    sentences = text.replace("!", ".").replace("?", ".").split(".")  # Normalize endings
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    return ". ".join(sentences) + "." if sentences else ""

# Loop through all JSON files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".json"):  # Process only JSON files
        file_path = os.path.join(input_dir, file)

        try:
            # Load JSON file
            df = pd.read_json(file_path, lines=True)

            # Filter valid articles
            df_filtered = df.query("status == 200").dropna(subset=["text"])

            # Sample 10 articles (or all if fewer than 10)
            sample = df_filtered[["url", "text"]].copy()

            # Extract only the first 5 sentences
            sample["text"] = sample["text"].apply(get_first_sentences)

            # Add source filename for reference
            sample["source_file"] = file

            # Append to list
            sample_records = sample.to_dict(orient="records")

            for i, record in enumerate(sample_records, start=len(all_samples)):
                record["id"] = i  # Assign a unique ID based on the current length of all_samples

            # Append to list
            all_samples.extend(sample_records)

            print(f"Processed {file}, sampled {len(sample)} articles.")

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Save all collected samples to a single JSON file
if all_samples:
    with open(output_filename, "w", encoding="utf-8") as f:
        for article in all_samples:
            f.write(f"{json.dumps(article, ensure_ascii=False)}\n")  # Write in JSON Lines format

    print(f"Saved {len(all_samples)} articles to {output_filename}.")
else:
    print("No valid articles found.")
