from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils.io import save_vocab_and_merge
from pathlib import Path

if __name__ == '__main__':
    source_file_name = 'owt_train'
    vocab_size = 32000

    corpus_path = Path(__file__).parent.parent / 'corpus-samples' / f'{source_file_name}.txt'
    result = train_bpe(corpus_path, vocab_size=vocab_size, special_tokens=["<|endoftext|>"])

    # store results in a file
    folder_path = Path(__file__).parent.parent / 'bpe_mappings'
    folder_path.mkdir(parents=True, exist_ok=True)

    save_vocab_and_merge(result[0], result[1], folder_path / f'{source_file_name}_{vocab_size}_vocab.json', folder_path / f'{source_file_name}_{vocab_size}_merges.txt')
    print("Training complete. Vocab and merges saved.")