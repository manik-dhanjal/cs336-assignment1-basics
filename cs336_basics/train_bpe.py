import os
from typing import BinaryIO
import regex as re
from multiprocessing import Process, Manager, Lock
from pathlib import Path
from collections import defaultdict
import time

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 200 * 1024 * 1024  # Read ahead by 200MB at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def preTokenizeChunk(filePath: str | os.PathLike, start: int, end: int, special_tokens: list[str], preTokenMap: dict[tuple[int, ...], int], lock):
    print('Child Process Started', start, end)
    mini_chunk_size = 200 * 1024 * 1024  # Read ahead by 200MB at a time
    currentPtr = start
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Open in binary mode for accurate seeking and reading
    with open(filePath, 'rb') as file:
        file.seek(start)
        while currentPtr <= end:
            to_read = min(mini_chunk_size, end - currentPtr)
            mini_chunk = file.read(to_read)
            if not mini_chunk:
                break
                
            # Decode only the bytes read, ignore errors to avoid decode issues at chunk boundaries
            text = mini_chunk.decode('utf-8', errors='ignore')
            
            # Split on special tokens (escaped for regex)
            split_regex = "|".join(re.escape(s) for s in special_tokens)
            segments = re.split(split_regex, text)
            local_token_map = {}
            for segment in segments:
                tokens = re.finditer(PAT, segment)
                for token_1 in tokens:
                    token = token_1.group()
                    word_in_int = tuple(map(int,token.encode('utf-8')))
                    local_token_map[word_in_int] = local_token_map.get(word_in_int, 0) + 1

            # Update the shared preTokenMap in one go
            lock.acquire()
            for k, v in local_token_map.items():
                preTokenMap[k] = preTokenMap.get(k, 0) + v
            lock.release()
            currentPtr += to_read

def merge(words: dict[list[int], int], pair: tuple[int, int], new_index: int) -> dict[list[int], int]:
    """
    Merge the most common pair of tokens in the words dictionary.
    """
    new_words = defaultdict(int);
    for word, count in words.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_index)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words[tuple(new_word)] = count
    return new_words

def get_most_frequent_pair(
    vocab: dict[int, bytes],
    pair_freqs: dict[tuple([int, int]), int]
) -> tuple[int, int]:
    """
    Find the most frequent pair of tokens in the words which is also lexicographically greatest.
    """
    max_freq = max(pair_freqs.values())
    max_pairs = {pair for pair, freq in pair_freqs.items() if freq == max_freq}
    return max(max_pairs, key=lambda x: (vocab[x[0]], vocab[x[1]]))  # Return the lexicographically greatest pair

def _update_byte_tuple(byte_tuple: tuple[int], merge_loc: int, new_index: int) -> tuple[tuple[int], tuple[int], tuple[int]]:
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix+( new_index,)+ suffix
    return new_byte_tuple, prefix, suffix

# create a function to print the values of count after converting key in binary of tuples:
def print_count(count: dict[tuple[int, int], int], vocab) -> None:
    """
    Convert the keys of the count dictionary from tuples of integers to tuples of bytes.
    """
    print({(vocab[k[0]], vocab[k[1]]): v for k, v in count.items()})

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    print(input_path, vocab_size, special_tokens)
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    chunk_start = time.time()
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    chunk_end = time.time()
    print(f"Time taken to calculate chunk boundaries: {chunk_end - chunk_start:.2f} seconds")

    pretoken_start = time.time()  # Start timing pre-tokenization
    # Shared dictionary and lock for token counting
    manager = Manager()
    preTokenMap: dict[tuple[int, ...], int] = manager.dict()
    lock = Lock()
    processInstance = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        p = Process(
            target=preTokenizeChunk,
            args=(input_path, start, end, special_tokens, preTokenMap, lock)
        )
        p.start()
        processInstance.append(p)

    for p in processInstance:
        p.join()

    start = time.time()
    num_merges = vocab_size - 256 - len(special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    vocab.update({
        256 + i: special.encode("utf-8") for i, special in enumerate(special_tokens)
    })
                
    converted_tokens = preTokenMap

     # count the frequency of pairs of tokens
    count: dict[tuple[int, int], int] = defaultdict(int)
    for word, appearances in converted_tokens.items():
        if len(word) < 2: 
            continue
        for index1, index2 in zip(word, word[1:]):
            count[(index1, index2)] = count.get((index1, index2), 0) + appearances
    
    for merge_num in range(num_merges):
        # Find the most frequent pair Lexicographically greatest pair
        freq_pair = get_most_frequent_pair(vocab, count)
        index1, index2 = freq_pair

        # merge the most common pair
        new_index = 256 + len(special_tokens) + merge_num
        
        merges.append((vocab[index1], vocab[index2]))
        vocab[new_index] = vocab[index1] + vocab[index2]
        
        
        new_pretoken_freq = {}
        for word, appearances in converted_tokens.items():
            if len(word) < 2:
                continue
            i = 0
            while i < len(word):
                pair = word[i:i+2]
                if pair == (index1, index2):
                    word, prefix, suffix = _update_byte_tuple(word, i, new_index)
                    
                    if prefix:
                        add_pair = (prefix[-1], new_index)
                        count[add_pair] = count.get(add_pair, 0) + appearances
                        delete_pair = (prefix[-1], index1)
                        count[delete_pair] = count.get(delete_pair, 0) - appearances
                    if suffix:
                        add_pair = (new_index, suffix[0])
                        count[add_pair] = count.get(add_pair, 0) + appearances
                        delete_pair = (index2, suffix[0])
                        count[delete_pair] = count.get(delete_pair, 0) - appearances
                    count[freq_pair] -= appearances
                i+=1
            new_pretoken_freq[word] = appearances
        converted_tokens = new_pretoken_freq
    end = time.time()
    print(f"Time taken for BPE merges: {end - start:.2f} seconds")

    return vocab, merges

if __name__ == '__main__':
    path = Path(__file__).parent.parent
    path = path / 'data'/'owt_valid.txt'
    path = 'data/small.txt'
    result = train_bpe(path, vocab_size=270, special_tokens=["<|endoftext|>"])
    print("Vocabulary:", result[0])
    print("Merges:", result[1])