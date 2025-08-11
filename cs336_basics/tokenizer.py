import base64
import os
import re
import collections
from collections import defaultdict
from itertools import pairwise
from typing import List, Tuple, Dict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class BPEtokenizer():
    def __init__(self, vocab={}, merges=[], special_tokens=[]):
        if vocab == {}:
            self.vocab = {i: bytes([i]) for i in range(256)}
            self.token_to_id = {bytes([i]): i for i in range(256)}
            self.merges = []
        else:
            self.vocab = vocab
            self.merges = merges
            self.token_to_id = {val: i for i, val in self.vocab.items()}
        self.special_tokens = special_tokens

    def chunked_data(input_path, num_processes, special_token=b"<|endoftext|>"):
        # parallel logic 
        chunks = []
        with open(input_path, "rb") as f: # type: ignore
            num_processes = 4
            boundaries = find_chunk_boundaries(f, desired_num_chunks=num_processes, split_special_token=special_token)

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _load_vocab(file_path: str | os.PathLike) -> Dict[int, bytes]:
        vocab: Dict[int, bytes] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue                   # skip header / blank lines
                idx_str, tok_b64 = line.rstrip("\n").split("\t")
                idx = int(idx_str)
                vocab[idx] = base64.b64decode(tok_b64)
        return vocab
    
    @staticmethod
    def _load_merges(file_path: str | os.PathLike) -> List[Tuple[bytes, bytes]]:
        merges: List[Tuple[bytes, bytes]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                t1_b64, t2_b64 = line.rstrip("\n").split(" ")
                merges.append(
                    (base64.b64decode(t1_b64), base64.b64decode(t2_b64))
                )
        return merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens if self.special_tokens is not None else []
        merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if special_tokens:
            sorted_specials = sorted(special_tokens, key=len, reverse=True)
            escaped = [re.escape(tok) for tok in sorted_specials]
            split_pat = f"({'|'.join(escaped)})"
            chunks = re.split(split_pat, text)
        else:
            chunks = [text]

        token_ids = []
        after_special = False  # NEW: track if last chunk was a special token

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_tokens:
                token_ids.append(self.token_to_id[chunk.encode("utf-8")])
                after_special = True
                continue

            # Byte-level tokens
            tokens = []
            for match in re.finditer(PAT, chunk):
                tokens.extend([bytes([b]) for b in match.group(0).encode("utf-8")])

            if after_special and len(tokens) >= 2 and tokens[0] == b'\n' and tokens[1] == b'\n':
                # Check raw chunk to see what follows the two newlines
                # If next char is non-whitespace -> keep separate; else allow merges.
                if re.match(r'^\n\n(?=\S)', chunk):
                    # emit two separate newline tokens, continue merging the remainder
                    token_ids.append(self.token_to_id[b'\n'])
                    token_ids.append(self.token_to_id[b'\n'])
                    tokens = tokens[2:]
                # else: do nothing here, let merge loop attempt to merge b'\n', b'\n' as usual


            # Apply merges
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                ranked_pairs = [(merge_ranks.get(pair, float('inf')), pair) for pair in pairs]
                best_rank, best_pair = min(ranked_pairs, key=lambda x: x[0])

                if best_rank == float('inf'):
                    break

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            token_ids.extend([self.token_to_id[tok] for tok in tokens])
            after_special = False  # reset flag after processing chunk

        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back into the original string.
        """
        # Convert each id to its byte sequence
        tokens_bytes = [self.vocab[i] for i in ids]

        # Concatenate all byte sequences
        text_bytes = b"".join(tokens_bytes)

        # Decode to string
        return text_bytes.decode("utf-8", errors="replace")


    def train(self, input_path: str, vocab_size: int = 259, special_tokens: list[str] = []):
        # load file 
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read(1024 * 1024 * 1000)

        # split on the special tokens 
        for spl_token in special_tokens:
            byte_token = spl_token.encode("utf-8") # byte token
            if byte_token in self.token_to_id:
                continue 
            curr_idx = len(self.vocab)
            self.vocab[curr_idx] = byte_token
            self.token_to_id[byte_token] = curr_idx

        # chunk the text    
        text_chunks = [text]
        for chunk in text_chunks:
            new_chunks = []
            for spl_token in special_tokens:
                # byte_token = spl_token.encode("utf-8") # byte token
                new_chunks.extend(chunk.split(spl_token))
            text_chunks = new_chunks

        # word stats 
        # escaped = [re.escape(tok) for tok in special_tokens]
        # spl_pattern = f"{"|".join(escaped)}"
        # text_chunks = re.split(spl_pattern, text)
        word_count = defaultdict(int)
        for chunk in text_chunks:
            for match in re.finditer(PAT, chunk):
                word_count[tuple([bytes([b]) for b in match.group(0).encode("utf-8")])] += 1
        n_train = vocab_size - len(self.vocab)
        # print(n_train)
        for _ in tqdm(range(n_train)):
            # create pair stat
            pair_stats = defaultdict(int)
            for word, count in word_count.items():
                for w1, w2 in pairwise(word):
                    pair_stats[(w1, w2)] += count           
            if not pair_stats:
                continue
            # best pair
            # get max_count and merge this pair
            best_pair, count = max(pair_stats.items(), key = lambda x: (x[1], x[0]))
            # self.merges.append(pair)
            # new_token = pair[0] + pair[1]
            new_token = best_pair[0] + best_pair[1]
            if not best_pair: continue  # empty   
            if new_token in self.vocab: continue # already seen
            self.merges.append(best_pair)
            curr_idx = len(self.vocab)
            self.vocab[curr_idx] = new_token
            self.token_to_id[new_token] = curr_idx

            # combine 
            new_word_count = Counter()
            for word, count in word_count.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_count[tuple(new_word)] += count
            # print(word_count)
            # print(new_word_count)
            word_count = new_word_count
            # break
        return self.vocab, self.merges, 
    
    from collections.abc import Iterable, Iterator
    def encode_iterable(self, iterable):
        """
        Lazily yield token IDs from an iterable of strings (e.g., file lines).
        Useful for large files that can't be loaded fully into memory.
        """
        special_tokens = self.special_tokens if self.special_tokens is not None else []
        merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        for text in iterable:
            if special_tokens:
                escaped = [re.escape(tok) for tok in special_tokens]
                split_pat = f"({'|'.join(escaped)})"
                chunks = re.split(split_pat, text)
            else:
                chunks = [text]

            for chunk in chunks:
                if not chunk:
                    continue

                if chunk in special_tokens:
                    yield self.token_to_id[chunk.encode("utf-8")]
                    continue

                # Byte-level tokenization
                tokens = []
                for match in re.finditer(PAT, chunk):
                    tokens.extend(tuple([bytes([b]) for b in match.group(0).encode("utf-8")]))

                # Apply merges
                while len(tokens) > 1:
                    pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                    ranked_pairs = [(merge_ranks.get(pair, float('inf')), pair) for pair in pairs]
                    best_rank, best_pair = min(ranked_pairs, key=lambda x: x[0])

                    if best_rank == float('inf'):
                        break

                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                            new_tokens.append(tokens[i] + tokens[i + 1])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens

                for tok in tokens:
                    yield self.token_to_id[tok]


if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    vocab, merges = tokenizer.train("./data/test_data.txt", vocab_size=260, special_tokens=["<|endoftext|>"])
    # n_vocab, n_merges = tokenizer.from_files("")
    print(vocab, "\n\n")
    print(merges, "\n\n")
