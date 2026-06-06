import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.id_to_word.update({0: self.pad_token, 1: self.unk_token, 2: self.bos_token, 3:self.eos_token})
        self.word_to_id.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})
        words = []
        for text in texts:
            words.extend(text.split(" "))
        i = 4
        for word in sorted(words):
            if word not in self.word_to_id.keys():
                self.word_to_id[word] = i
                self.id_to_word[i] = word
                i+=1
        self.vocab_size = len(self.word_to_id.keys())
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if not text.strip():
            return []
        return [self.word_to_id.get(word.lower().strip(), 1) for word in text.strip().split(" ")]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        if not ids:
            return ""
        return " ".join(self.id_to_word.get(id, self.unk_token) for id in ids)
