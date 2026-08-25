import random
import os
import json
import re
import math
import unicodedata
import numpy as np
from transformers import AutoTokenizer


def apply_attack(token_list: list, attack_type: str, rate: float, tokenizer: "AutoTokenizer") -> list:

    if attack_type == "edit":
        vocab_size = tokenizer.vocab_size
        return edit_attack(token_list, rate, vocab_size)

    elif attack_type == "homoglyph":
        text = tokenizer.decode(token_list)
        attacked_text = homoglyph_attack(text, rate)
        return tokenizer.encode(attacked_text)
    
    elif attack_type == "invisible_char":
        text = tokenizer.decode(token_list)
        attacked_text = invisible_char_attack(text, rate)
        return tokenizer.encode(attacked_text)
    
    elif attack_type == "synonym":
        text = tokenizer.decode(token_list)
        attacked_text = synonym_attack(text, rate)
        return tokenizer.encode(attacked_text)
    
    elif attack_type == "token_ambiguity":
        return token_ambiguity(token_list, tokenizer)

    elif attack_type == "social_platform":
        text = tokenizer.decode(token_list)
        attacked_text = social_platform_sanitize(text)
        return tokenizer.encode(attacked_text)

    elif attack_type == "mobile_im":
        text = tokenizer.decode(token_list)
        attacked_text = mobile_im_normalize(text)
        return tokenizer.encode(attacked_text)

    elif attack_type == "document_reformat":
        text = tokenizer.decode(token_list)
        attacked_text = document_reformat(text)
        return tokenizer.encode(attacked_text)

    else:
        return token_list  # No attack applied

def edit_distance(s1: list, s2: list) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        prev_diag = previous_row[0]
        previous_row[0] = i + 1
        
        for j, c2 in enumerate(s2):
            old_val = previous_row[j+1]
            
            if c1 == c2:
                previous_row[j+1] = prev_diag
            else:
                
                v1 = old_val + 1
                v2 = previous_row[j] + 1
                v3 = prev_diag + 1
                
                if v1 <= v2:
                    previous_row[j+1] = v1 if v1 <= v3 else v3
                else:
                    previous_row[j+1] = v2 if v2 <= v3 else v3
            
            prev_diag = old_val
            
    return previous_row[-1]

def edit_attack(token_list: list[int], error_rate: float, vocab_size: int = 50257) -> list[int]:
    if not token_list or error_rate <= 0:
        return token_list.copy()

    n = len(token_list)

    edit_prob = np.random.random(n)
    edit_type = np.random.randint(0, 3, n)
    random_token = np.random.randint(0, vocab_size, n)

    result = []

    for i in range(n):
        if edit_prob[i] < error_rate:
            if edit_type[i] == 0:

                new_token = random_token[i]
                if new_token == token_list[i] and vocab_size > 1:
                    new_token = (new_token + 1) % vocab_size
                result.append(new_token)
            elif edit_type[i] == 1:
                continue
            else:
                result.append(token_list[i])
                result.append(random_token[i])
        else:
            result.append(token_list[i])

    return result



HOMOGLYPHS = {
    'a': ['а', 'ａ'],
    'c': ['с', 'ｃ'],
    'e': ['е', 'ｅ'],
    'o': ['о', 'ｏ'],
    'p': ['р', 'ｐ'],
    'x': ['х', 'ｘ'],
    'y': ['у', 'ｙ'],
    'A': ['Α', 'А', 'Ａ'],
    'B': ['Β', 'В', 'Ｂ'],
    'C': ['С', 'Ｃ'],
    'E': ['Ε', 'Е', 'Ｅ'],
    'H': ['Η', 'Н', 'Ｈ'],
    'I': ['Ι', 'І', 'Ｉ'],
    'K': ['Κ', 'К', 'Ｋ'],
    'M': ['Μ', 'М', 'Ｍ'],
    'N': ['Ν', 'Ｎ'],
    'O': ['Ο', 'О', 'Ｏ'],
    'P': ['Ρ', 'Р', 'Ｐ'],
    'T': ['Τ', 'Т', 'Ｔ'],
    'X': ['Χ', 'Х', 'Ｘ'],
    'Y': ['Υ', 'Ү', 'Ｙ'],
    'Z': ['Ζ', 'Ｚ'],
}

def homoglyph_attack(text: str, rate: float = 0.1) -> str:
    parts = re.split(r'(\s+)', text)
    modified_parts = []
    for part in parts:
        if part and not part.isspace():
            if random.random() < rate:
                candidates = [i for i, c in enumerate(part) if c in HOMOGLYPHS]
                if candidates:
                    idx = random.choice(candidates)
                    chars = list(part)
                    chars[idx] = random.choice(HOMOGLYPHS[chars[idx]])
                    modified_parts.append("".join(chars))
                else:
                    modified_parts.append(part)
            else:
                modified_parts.append(part)
        else:
            modified_parts.append(part) 
    return "".join(modified_parts)


INVISIBLE_CHAR = [
    '\n', '\t', '\r', '\v', '\f'
]

def invisible_char_attack(text: str, rate: float = 0.1) -> str:
    parts = re.split(r'(\s+)', text)
    modified_parts = []
    
    for part in parts:
        if part and not part.isspace():
            if random.random() < rate:
                modified_parts.append(part + random.choice(INVISIBLE_CHAR))
            else:
                modified_parts.append(part)
        else:
            modified_parts.append(part)
            
    return "".join(modified_parts)

# Load synonyms from the generated JSON file
SYNONYMS_DB = {}
try:
    # Try multiple locations for synonyms.json
    search_paths = [
        os.path.join(os.getcwd(), 'data', 'synonyms.json'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'synonyms.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'synonyms.json'),
    ]
    json_path = None
    for p in search_paths:
        if os.path.exists(p):
            json_path = p
            break
    if json_path:
        with open(json_path, 'r') as f:
            SYNONYMS_DB = json.load(f)
    else:
        print(f"synonyms.json not found in any of: {search_paths}")
        raise FileNotFoundError("synonyms.json not found.")
except Exception as e:
    raise RuntimeError(f"Critical Error: Could not load synonyms database. {e}")

def synonym_attack(text: str, rate: float = 0.1) -> str:

    parts = re.split(r'(\s+)', text)
    modified_parts = []
    
    for part in parts:
        if part and not part.isspace():
            word = part
            # Simple punctuation handling
            clean_word = word.strip(".,!?;:\"'")
            if not clean_word:
                modified_parts.append(word)
                continue
                
            prefix = word[:word.find(clean_word)] if clean_word in word else ""
            suffix = word[word.find(clean_word)+len(clean_word):] if clean_word in word else ""
            
            lower_word = clean_word.lower()
            synonyms = []
            
            # Try loaded JSON DB
            if lower_word in SYNONYMS_DB:
                synonyms = SYNONYMS_DB[lower_word]
                
            if synonyms and random.random() < rate:
                replacement = random.choice(synonyms)
                # Match case
                if clean_word.istitle():
                    replacement = replacement.capitalize()
                elif clean_word.isupper():
                    replacement = replacement.upper()
                
                modified_parts.append(prefix + replacement + suffix)
            else:
                modified_parts.append(word)
        else:
            modified_parts.append(part)
            
    return "".join(modified_parts)

def token_ambiguity(token_list: list, tokenizer: "AutoTokenizer") -> list:
    
    text = tokenizer.decode(token_list, skip_special_tokens=False)
    encoded_token_list = tokenizer.encode(text)
    if encoded_token_list != token_list:
        return encoded_token_list
    else:
        # if not, try to create ambiguity by inserting spaces randomly
        chars = np.array(list(text))
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            mask = np.random.rand(len(chars)) < 0.01 # For token ambiguity is not a high probability task
            if not np.any(mask):
                attempts += 1
                continue
            spaces = np.full(len(chars), '', dtype='<U1')
            spaces[mask] = ' '
            interleaved = np.char.add(chars, spaces)
            modified_text = ''.join(interleaved)
            new_encoded = tokenizer.encode(modified_text)
            if new_encoded != token_list:
                return new_encoded
            attempts += 1
        return token_list  # Fallback to original if no ambiguity created


# ── Real-world channel perturbations ──────────────────────────────────────

_speller = None


def _get_speller():
    global _speller
    if _speller is None:
        from autocorrect import Speller
        _speller = Speller(lang="en")
    return _speller


def social_platform_sanitize(text: str) -> str:
    """Simulate social-platform sanitization (NFKC, strip markup, collapse ws)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[*_~`#>-]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = "".join(c for c in text if c.isprintable())
    text = " ".join(text.split())
    return text


def mobile_im_normalize(text: str) -> str:
    """Simulate mobile-IM autocorrect (spelling, quotes, emoji, punctuation)."""
    text = _get_speller()(text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    try:
        import emoji as _emoji
        text = _emoji.replace_emoji(text, replace="")
    except ImportError:
        pass
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    return text


def document_reformat(text: str) -> str:
    """Simulate document reformat / OCR corruption (NFKD, drop non-ASCII)."""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[–—]", "-", text)
    return text

        