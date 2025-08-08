import re
import collections
from collections import defaultdict
from itertools import pairwise
from typing import List, Tuple, Dict
from cs336_basics.pretokenization_example import find_chunk_boundaries

class BPEtokenizer():
    def __init__(self):
        self.special_tokens = []
        self.pair_counts = collections.defaultdict(int)
        self.pair_to_words = collections.defaultdict(list) # multiple words 
        self.vocab  = {i: bytes(i) for i in range(256)}
        self.merges = []
        # self.splitter = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.splitter_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+""")
    
    def train(self, input_path: str, vocab_size: int = 259, special_tokens: list[str] = []):
        # update special tokens
        self.special_tokens = special_tokens
        for token_str in self.special_tokens:
            if token_str.encode("utf-8") not in self.vocab:
                self.vocab[len(self.vocab)] = token_str.encode("utf-8")

        # Initilize vocab 
        word_counts = self.load_data(input_path)
        self.word_counts = word_counts
        
        # print(len(self.word_counts))
        for word, count in word_counts.items():
            for c1, c2 in pairwise(word):
                self.pair_counts[(c1, c2)] += count
                self.pair_to_words[(c1, c2)].append(word)

        # Perform BPE merges
        initial_vocab_size = len(self.vocab)
        num_merges = vocab_size - initial_vocab_size

        for i in range(num_merges):
            self.merge()
        
        # self.vocab = {k: v for k, v in self.vocab.items()}

        def is_valid_token(b: bytes) -> bool:
            if not b:
                return False  # b''
            if all(byte == 0 for byte in b):
                return False  # b'\x00\x00...'
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                return False
            return True

        # ONLY clean tokens **added after** base vocab
        self.vocab = {
            k: v
            for k, v in self.vocab.items()
        }
        return self.vocab, self.merges

    def process_chunk(self, input_path, start, end):
        local_word_counts = collections.defaultdict(int)
        # load chunk and create word statistics for local chunk
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors='replace')

            # each chunk should split by special characters
            text_parts = [chunk_text]
            for token_str in self.special_tokens:
                new_chunk_text = []
                for parts in text_parts:
                    new_chunk_text.extend(parts.split(token_str))
                text_parts = new_chunk_text

            # each chunk is text between special token 
            for sub_chunk in text_parts:
                if not sub_chunk:
                    continue
                p_iter = self.splitter_pattern.finditer(sub_chunk)
                for match in p_iter:
                    word_bytes = match.group().encode("utf-8")
                    word_tuple = tuple([bytes([b]) for b in word_bytes])
                    local_word_counts[word_tuple] += 1
                    # word_tuple = tuple([bytes([b]) for b in match.group().encode("utf-8")])
                    # local_word_counts[word_tuple] += 1
        return local_word_counts


    def load_data(self, input_path):
        chunks = []
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>") # start, end 
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk = self.process_chunk(input_path, start, end)
            chunks.append(chunk)
        word_counts = defaultdict(int)
        for local_dict in chunks:
            for word, count in local_dict.items():
                word_counts[word] += count
        return word_counts

    def merge(self):
        if not self.pair_counts:
            return

        # Find most frequent pair
        most_frequent_pair = max(
                self.pair_counts.items(),
                key=lambda x: (x[1], x[0])  # frequency, then lexicographic tie-break
            )[0]
        pair_freq = self.pair_counts[most_frequent_pair]

        if not most_frequent_pair[0] or not most_frequent_pair[1]:
            return  # Skip invalid merge

        # print(f"Merging pair: {most_frequent_pair} (freq: {pair_freq})")
        merged_symbol = most_frequent_pair[0] + most_frequent_pair[1]
        merged_token = tuple([merged_symbol])

        new_word_counts = collections.defaultdict(int)

        for word, count in self.word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if current and next token match the pair
                if i < len(word) - 1 and (word[i], word[i+1]) == most_frequent_pair:
                    new_word.append(merged_symbol)
                    i += 2  # Skip the next symbol (it's part of the merged pair)
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] += count
        
        # Replace word counts with updated words
        self.word_counts = new_word_counts

        # Reset pair counts and pair-to-words map
        self.pair_counts = collections.defaultdict(int)
        self.pair_to_words = collections.defaultdict(list)

        for word, count in self.word_counts.items():
            for c1, c2 in pairwise(word):
                self.pair_counts[(c1, c2)] += count
                self.pair_to_words[(c1, c2)].append(word)

        # Track the merge
        self.merges.append(most_frequent_pair)

        # Add to vocab if not already in
        if merged_symbol not in self.vocab.values():
            self.vocab[len(self.vocab)] = merged_symbol

if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    vocab, merges = tokenizer.train("./data/test_data.txt", vocab_size=260, special_tokens=["<|endoftext|>"])
    print(vocab, "\n\n")
    print(merges, "\n\n")
