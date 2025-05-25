import json
import re
import unicodedata
from typing import List, Tuple


class BpeTokenizer:
    def __init__(self, tokenizer_json_path: str):
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer = json.load(f)

        # Load vocab
        model = tokenizer["model"]
        self.vocab = model["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Load merges
        self.merges = [tuple(m) for m in model["merges"]]
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Special tokens
        self.special_tokens = {
            t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])
        }

        # Normalization
        self.nfc_normalize = tokenizer.get("normalizer", {}).get("type") == "NFC"

        # Pre-tokenization regex
        self.pattern = re.compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = []
        if self.nfc_normalize:
            text = unicodedata.normalize("NFC", text)

        if add_special_tokens:
            if "<|im_start|>" in self.special_tokens:
                tokens.append(self.special_tokens["<|im_start|>"])

        for token in self._pre_tokenize(text):
            if token in self.special_tokens:
                tokens.append(self.special_tokens[token])
            else:
                tokens.extend(
                    self.vocab[bpe_token]
                    for bpe_token in self._bpe(token)
                    if bpe_token in self.vocab
                )

        if add_special_tokens:
            if "<|im_end|>" in self.special_tokens:
                tokens.append(self.special_tokens["<|im_end|>"])

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for id in token_ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                special = next((k for k, v in self.special_tokens.items() if v == id), "<unk>")
                tokens.append(special)

        raw = ''.join(tokens)
        cleaned = raw.replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ")
        return unicodedata.normalize("NFC", cleaned) if self.nfc_normalize else cleaned

    def _pre_tokenize(self, text: str) -> List[str]:
        return self.pattern.findall(text)

    def _get_pairs(self, word: List[str]) -> set:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token: str) -> List[str]:
        word = list(token)
        pairs = self._get_pairs(word)
        while pairs:
            best = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if best not in self.bpe_ranks:
                break

            first, second = best
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
        return word
'''

# Save to file
output_path = Path("/mnt/data/bpe_tokenizer.py")
output_path.write_text(python_tokenizer_code)

output_path.name
Result
'bpe_tokenizer.py'
✅ The Python version of your BpeTokenizer class has been created.

📄 Download bpe_tokenizer.py

It supports:

Loading tokenizer.json from any path

BPE merges

Special token handling

NFC normalization

Let me know if you’d like a test script or Jupyter demo for it. 






You said:
add comments please , and like a normal person


ChatGPT said:

Analyzed
python
Always show details

Copy
# Add comments to the tokenizer code
commented_code = '''
import json
import re
import unicodedata
from typing import List, Tuple


class BpeTokenizer:
    def __init__(self, tokenizer_json_path: str):
        # Load tokenizer.json file
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer = json.load(f)

        # Load vocab: string token -> int ID
        model = tokenizer["model"]
        self.vocab = model["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}  # inverse map

        # Load BPE merge rules and assign ranking
        self.merges = [tuple(m) for m in model["merges"]]
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Load special tokens (e.g., <|im_start|>, <|im_end|>)
        self.special_tokens = {
            t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])
        }

        # Check if normalization is required
        self.nfc_normalize = tokenizer.get("normalizer", {}).get("type") == "NFC"

        # Pre-tokenization regex based on Hugging Face tokenizer standards
        self.pattern = re.compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a string into a list of token IDs.
        Optionally includes special tokens like <|im_start|> and <|im_end|>.
        """
        tokens = []

        if self.nfc_normalize:
            text = unicodedata.normalize("NFC", text)

        if add_special_tokens:
            if "<|im_start|>" in self.special_tokens:
                tokens.append(self.special_tokens["<|im_start|>"])

        for token in self._pre_tokenize(text):
            if token in self.special_tokens:
                tokens.append(self.special_tokens[token])
            else:
                # Apply BPE to split token into subwords
                tokens.extend(
                    self.vocab[bpe_token]
                    for bpe_token in self._bpe(token)
                    if bpe_token in self.vocab
                )

        if add_special_tokens:
            if "<|im_end|>" in self.special_tokens:
                tokens.append(self.special_tokens["<|im_end|>"])

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back into a string.
        """
        tokens = []
        for id in token_ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                # Try to decode as a special token
                special = next((k for k, v in self.special_tokens.items() if v == id), "<unk>")
                tokens.append(special)

        raw = ''.join(tokens)
        cleaned = raw.replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ")
        return unicodedata.normalize("NFC", cleaned) if self.nfc_normalize else cleaned

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Splits input string into pre-tokens based on regex.
        """
        return self.pattern.findall(text)

    def _get_pairs(self, word: List[str]) -> set:
        """
        Get set of symbol pairs in a word.
        """
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token: str) -> List[str]:
        """
        Apply BPE algorithm to a single token.
        """
        word = list(token)
        pairs = self._get_pairs(word)
        while pairs:
            best = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if best not in self.bpe_ranks:
                break

            first, second = best
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
        return word


tokenizer = BpeTokenizer(tokenizer_json_path=r"C:\Others\Projects\hf_optimum\qweb_3_0.6B_onnx_from_HF\tokenizer.json")
prompt = "Who are you?"
tokens = tokenizer.tokenize(prompt)
print(tokens)
decoded = tokenizer.decode(tokens)
print(decoded)
