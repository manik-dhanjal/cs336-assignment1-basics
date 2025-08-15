from pathlib import Path

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        pass

    def train(self, corpus_path: str|Path, vocab_size: int, special_tokens: list[str] = ["<|endoftext|>"]):
        pass

    def encode(self, text: str) -> list[int]:
        pass
    
    def decode(self, tokens: list[int]) -> str:
        pass