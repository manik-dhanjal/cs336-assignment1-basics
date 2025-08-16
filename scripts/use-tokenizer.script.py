
from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files("tests/fixtures/gpt2_vocab.json", "tests/fixtures/gpt2_merges.txt", special_tokens=["<|endoftext|>"])

text =  "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
print(text)
encoded_tokens = tokenizer.encode(text)
print(encoded_tokens)
decoded_text = tokenizer.decode(encoded_tokens)
print(decoded_text)