
from cs336_basics.tokenizer import Tokenizer

def get_compression_ratio(data_set_name: str, vocab_size: int, text: str|None = None, print_results: bool = False) -> float:
    tokenizer = Tokenizer.from_files(f"bpe_mappings/{data_set_name}_{vocab_size}_vocab.json", f"bpe_mappings/{data_set_name}_{vocab_size}_merges.txt", special_tokens=["<|endoftext|>"])

    if text is None:
        text =  "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_tokens = tokenizer.encode(text, progress_bar=True)

    if print_results:
        # print(f"Original text: {text}")
        # print(f"Encoded tokens: {encoded_tokens}")
        decoded_text = tokenizer.decode(encoded_tokens, progress_bar=True)
        # print(f"Decoded text: {decoded_text}")
        print(f"are both original and decoded text same: {decoded_text == text}")
        print(f"Compression ratio (encoded tokens / original chars): {len(encoded_tokens) / len(text):.2f}")

    compression_ratio = len(encoded_tokens) / len(text)
    return compression_ratio


with open('corpus-samples/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    print(get_compression_ratio("owt_train", 10000, text, print_results=True))