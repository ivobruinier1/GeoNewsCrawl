import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from collections import Counter
import string

# Hardcoded punctuation marks to remove
PUNCTUATION = {":", ";", "-", "(", ")", "[", "]", "{", "}", "'", "\"", "$", "%", "&", "*", "@", "#", "_", "+", "="}


def get_word_distribution(data):
    words = [line.split()[0].lower() for line in data.strip().split("\n") if line]
    # Remove tokens that consist only of punctuation
    words = [word for word in words]
    word_counts = Counter(words)
    total = sum(word_counts.values())
    distribution = {word: count / total for word, count in word_counts.items()}

    print("\nWord Distribution:")
    for word, prob in list(distribution.items())[:10]:  # Show first 10 words for debugging
        print(f"{word}: {prob:.6f}")

    return word_counts, distribution

def plot_word_frequencies(word_counts1, word_counts2):
    common_words = set(word_counts1.keys()).intersection(set(word_counts2.keys()))
    shared_percentage = len(common_words) / len(set(word_counts1.keys()).union(set(word_counts2.keys()))) * 100
    print(f"Shared vocabulary percentage: {shared_percentage:.2f}%")

    top_words1 = word_counts1.most_common(10)
    top_words2 = word_counts2.most_common(10)

    print("\nTop 10 words in Dataset 1:")
    for word, count in top_words1:
        print(f"{word}: {count}")

    print("\nTop 10 words in Dataset 2:")
    for word, count in top_words2:
        print(f"{word}: {count}")

    words1, counts1 = zip(*top_words1)
    words2, counts2 = zip(*top_words2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(words1, counts1, color='blue')
    ax[0].set_title("Top Words in Dataset 1")
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].bar(words2, counts2, color='red')
    ax[1].set_title("Top Words in Dataset 2")
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def js_divergence(dataset1, dataset2):
    print("Loading dataset 1...")
    with open(dataset1, 'r', encoding='utf-8') as f:
        data1 = f.read()
    print("Dataset 1 loaded.")

    print("Loading dataset 2...")
    with open(dataset2, 'r', encoding='utf-8') as f:
        data2 = f.read()
    print("Dataset 2 loaded.")

    word_counts1, P = get_word_distribution(data1)
    word_counts2, Q = get_word_distribution(data2)

    all_words = set(P.keys()).union(set(Q.keys()))
    print(f"Total unique words: {len(all_words)}")

    P_arr = np.array([P.get(word, 0) for word in all_words])
    Q_arr = np.array([Q.get(word, 0) for word in all_words])

    print("First 10 P values:", P_arr[:10])
    print("First 10 Q values:", Q_arr[:10])

    js_score = jensenshannon(P_arr, Q_arr, base=2)
    print(f"Computed Jensen-Shannon Divergence: {js_score:.6f}")

    plot_word_frequencies(word_counts1, word_counts2)

    return js_score

# Example usage
dataset1 = "common_craw_news_test/us_news_train.tsv"  # Your first dataset
dataset2 = "common_craw_news_test/cc_news_test.tsv"  # Your second dataset

js_score = js_divergence(dataset1, dataset2)
print(f"Final JS Divergence Score: {js_score:.4f}")
