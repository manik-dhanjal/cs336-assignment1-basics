from pathlib import Path
import json
from cs336_basics.utils.io import gpt2_bytes_to_unicode, GPT2_PRETOKENIZER_PATTERN
import regex as re
from typing import Iterable
from tqdm import tqdm

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.int_to_byte = vocab
        self.byte_to_int = {v: k for k, v in vocab.items()}

        self.merges = {(self.byte_to_int[merge[0]], self.byte_to_int[merge[1]]): self.byte_to_int[b''.join(merge)] for merge in merges}
        
        self.special_tokens = None
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = special_tokens
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in self.byte_to_int:
                    self.byte_to_int[special_token.encode('utf-8')] = len(self.int_to_byte)
                    self.int_to_byte[len(self.int_to_byte)] = special_token.encode('utf-8')

    def _create_pairs(self, token: tuple[int,...]) -> set[tuple[int, int]]:
        pairs = set()
        for prev, post in zip(token, token[1:]):
            pairs.add((prev, post))
        return pairs
    
    def _merge_pair_in_token(self, token: tuple[int,...], pair: tuple[int,int], new_indicies: int) -> tuple[int,...]:
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token)-1 and (token[i], token[i+1]) == pair:
                new_token.append(new_indicies)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        return tuple(new_token)

    @classmethod
    def from_files(cls, vocab_filepath: str|Path, merges_filepath: str|Path, special_tokens: list[str]| None = None):
        unicode_to_byte = {v: bytes([k]) for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, 'r') as vocab_file:
            vocab_raw = json.load(vocab_file)
            vocab = {id: b''.join([unicode_to_byte[t] for t in token]) for token, id in vocab_raw.items()}

        with open(merges_filepath, 'r') as merges_file:
            raw_merges = [tuple(line.strip().split()) for line in merges_file]

            merges = [(b''.join([unicode_to_byte[a] for a in prefix]), b''.join([unicode_to_byte[b] for b in suffix])) for prefix, suffix in raw_merges]
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        if self.special_tokens and text in self.special_tokens:
            return [self.byte_to_int[text.encode('utf-8')]]
        else:
            words = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
            pretokens = [ [self.byte_to_int[bytes([b])] for b in word.encode('utf-8')] for word in words]
            encoded_tokens = []
            for pre_token in pretokens:
                pairs = self._create_pairs(pre_token)
                while len(pairs)>=1:
                    pair_to_merge = min(pairs, key = lambda pair: self.merges.get(pair, float('inf')))
                    if pair_to_merge not in self.merges:
                        break
                    pre_token = self._merge_pair_in_token(pre_token, pair_to_merge, self.merges[pair_to_merge])
                    pairs = self._create_pairs(pre_token)
                encoded_tokens.extend(pre_token)
            return encoded_tokens
    
    def encode(self, text: str, progress_bar: bool=False) -> list[int]:
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
            
        else:
            special_split_chunk = [text]
            
        ids = []
        for chunk in tqdm(special_split_chunk, disable=not progress_bar,
                          desc=f"Encoding {len(special_split_chunk)} documents"):
            ids += self._encode_chunk(chunk)
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id
    
    
    def decode(self, tokens: list[int]) -> str:
        tokens_in_bytes = b''.join([ self.int_to_byte[token] for token in tokens])
        return tokens_in_bytes.decode('utf-8', errors='replace')