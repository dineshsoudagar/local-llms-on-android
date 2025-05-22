# Simulate the execution of the user-supplied script using the uploaded tokenizer.json
# and display the print outputs step-by-step.
import json

# Use a shortened input for readability and demonstration.
test_text = "Hello world!"

# We'll wrap the script from earlier into a function and run it
def simulate_bpe_tokenizer(text: str, tokenizer):
    import re

    vocab = tokenizer["model"]["vocab"]
    merges = tokenizer["model"]["merges"]
    id_to_token = {v: k for k, v in vocab.items()}
    bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}

    # STEP 1: Pre-tokenization
    def pre_tokenize(text: str):
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        print(f"Pre-tokenized text: {tokens}")
        return tokens

    # STEP 2: Get all symbol pairs
    def get_pairs(word):
        return set((word[i], word[i + 1]) for i in range(len(word) - 1))

    # STEP 3: Apply BPE merging
    def bpe_encode_token(token):
        word = list(token)
        print(f"\nEncoding token: {token}")
        print(f"Initial chars: {word}")
        pairs = get_pairs(word)
        while pairs:
            best = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
            if best not in bpe_ranks:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            print(f"Merged {best} → {word}")
            pairs = get_pairs(word)
        return word

    # STEP 4: Encode
    def encode(text):
        tokens = pre_tokenize(text)
        token_ids = []
        for token in tokens:
            bpe_tokens = bpe_encode_token(token)
            for bpe in bpe_tokens:
                token_id = vocab.get(bpe, -1)
                token_ids.append(token_id)
                print(f"BPE '{bpe}' → Token ID: {token_id}")
        return token_ids

    # STEP 5: Decode
    def decode(token_ids):
        print("\nDecoding...")
        tokens = [id_to_token.get(i, "<unk>") for i in token_ids]
        print(f"Tokens: {tokens}")
        return "".join(tokens)

    print(f"Original Text: {text}")
    encoded_ids = encode(text)
    print(f"\nEncoded Token IDs: {encoded_ids}")
    decoded_text = decode(encoded_ids)
    print(f"\nDecoded Text: {decoded_text}")
    return encoded_ids, decoded_text

# Load the tokenizer from file
with open("checkpoints/Qwen/Qwen2.5-0.5B-Instruct/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Run the test
simulate_bpe_tokenizer("Hello world!", tokenizer_data)
