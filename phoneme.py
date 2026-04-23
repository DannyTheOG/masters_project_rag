import random

# Path to Mac system word list
word_file = "/usr/share/dict/words"

# Load words from the file
with open(word_file, "r") as f:
    words = f.read().splitlines()
    # print(words)
    all_words = [wrd.strip().lower() for wrd in words if wrd.isalpha()]
    
# Shuffle and select 1000 unique words
random.shuffle(all_words)
selected_words = all_words[:1000]

# Save to file
output_file = "english_words_1000.txt"
with open(output_file, "w") as f:
    for i, word in enumerate(selected_words, 1):
        f.write(f"{i}. {word}\n")

print(f"1000 English words saved to {output_file}")