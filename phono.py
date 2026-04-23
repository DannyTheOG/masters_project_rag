import random
import re

# Path to Mac system word list
word_file = "/usr/share/dict/words"

# Load words
with open(word_file, "r") as f:
    all_words = [word.strip().lower() for word in f if word.strip()]

# Keep words likely useful for phonological processes
# Short words, 2–6 letters, mostly consonant-vowel patterns
phonology_words = [
    word for word in all_words
    if 2 <= len(word) <= 4
    and re.fullmatch(r"[a-z]+", word)  # alphabetic only
]

# Shuffle and select 500
random.shuffle(phonology_words)
selected_words = phonology_words[:5000]

# Save to file
output_file = "phonological_words_5000.txt"
with open(output_file, "w") as f:
    for i, word in enumerate(selected_words, 1):
        f.write(f"{i}. {word}\n")

print(f"500 words for phonological processes saved to {output_file}")