import random
import csv

# Consonant phoneme contrasts commonly used in therapy
phoneme_pairs = [
    ("p","b"), ("t","d"), ("k","g"),
    ("f","v"), ("s","z"), ("s","sh"),
    ("l","r"), ("m","n"), ("ch","j"),
    ("th","f"), ("w","r")
]

# Word templates
onsets = ["p","b","t","d","k","g","f","v","s","z","l","r","m","n","sh","ch"]
vowels = ["a","e","i","o","u","ee","oo","ai","oa"]
codas = ["p","t","k","b","d","g","m","n","s","sh","ch","l","r"]

# Build base words
def generate_word():
    onset = random.choice(onsets)
    vowel = random.choice(vowels)
    coda = random.choice(codas)
    return onset + vowel + coda

# Replace phoneme
def create_minimal_pair(word, p1, p2):
    if word.startswith(p1):
        return word.replace(p1, p2, 1)
    elif word.startswith(p2):
        return word.replace(p2, p1, 1)
    return None

def generate_pairs(num_pairs=50000, output_file="phoneme_minimal_pairs.csv"):

    pairs = []
    attempts = 0

    while len(pairs) < num_pairs and attempts < num_pairs*5:
        attempts += 1

        word = generate_word()
        p1, p2 = random.choice(phoneme_pairs)

        if word.startswith(p1) or word.startswith(p2):
            pair_word = create_minimal_pair(word, p1, p2)

            if pair_word and pair_word != word:
                pairs.append((word, pair_word, f"{p1}-{p2}"))

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","word_A","word_B","phoneme_contrast"])

        for i,(a,b,c) in enumerate(pairs):
            writer.writerow([i+1,a,b,c])

    print(f"Generated {len(pairs)} minimal pairs")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    generate_pairs(5000)