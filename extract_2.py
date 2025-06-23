import json
import re
import csv
import time

input_jsonl = 'common_crawl_news.json'
output_tsv = 'common_crawl_news.tsv'

end_tokens = ["!", "?", "."]

def tokenize_text(text):
    text = re.sub(r"([.!?])(?=[ \n])", r"\1\n\n", text)
    tokens = re.findall(r"\w+|[.,!?]", text)
    return tokens

def process_json_to_tsv(input_jsonl, output_tsv, log_interval=1000):
    start_time = time.time()
    article_count = 0

    try:
        with open(input_jsonl, "r", encoding="utf-8") as json_file, \
             open(output_tsv, 'w', newline='', encoding='utf-8') as tsvfile:

            writer = csv.writer(tsvfile, delimiter='\t')

            for line in json_file:
                article = json.loads(line)
                article_count += 1

                if "text" in article and article["text"].strip():
                    article_id = article.get("id", "NA")
                    tokens = tokenize_text(article["text"])

                    for token in tokens:
                        if token in end_tokens:
                            writer.writerow([token, 'O', article_id])
                            writer.writerow([])  # Blank line
                        elif token.strip() == "":
                            writer.writerow([])  # Blank line
                        else:
                            writer.writerow([token, 'O', article_id])

                    writer.writerow([])  # Extra newline between articles

                if article_count % log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"[{article_count} articles] Time elapsed: {elapsed:.2f} seconds")

        total_time = time.time() - start_time
        print(f"✅ Done. TSV file written to: {output_tsv}")
        print(f"⏱️ Total time taken: {total_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error processing {input_jsonl}: {e}")

process_json_to_tsv(input_jsonl, output_tsv, log_interval=1000)
