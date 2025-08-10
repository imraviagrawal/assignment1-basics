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
# class BPEtokenizer():
#     def __init__(self, vocab={}, merges=[], special_tokens=None):
#         self.special_tokens = []
#         if vocab == {}:
#             self.vocab = {i: bytes([i]) for i in range(256)}
#             self.token_to_id = {bytes([i]): i for i in range(256)}
#         else:
#             self.vocab = vocab
#         self.merges = merges
#         self.word_count = Counter() # word count
#         self.special_tokens = special_tokens
    
#     def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#         # read and return vocab and merges
#         pass 


#     def encode(self, text: str) -> list[int]: 
#         # encode and return text output 
#         pass

#     def encode_iterable(self, iterable):
#         pass 

#     def train(self, input_path: str, vocab_size: int = 259, special_tokens: list[str] = []):
#         # read file 
#         # with open(input_path, "rb") as f:
#         with open(input_path, "r", encoding="utf-8") as f:
#             text = f.read(1024 * 1024 * 1000)

#         # add special token to the vocab lookup
#         for spl_token in special_tokens:
#             token_bytes = spl_token.encode("utf-8")
#             if token_bytes not in self.token_to_id:
#                 curr_idx = len(self.vocab)
#                 self.vocab[curr_idx] = token_bytes
#                 self.token_to_id[token_bytes] = curr_idx
        
#         # split text on special tokens, first split on special tokens,
#         text_chunk = [text]
#         for chunk in text_chunk:
#             new_chunks = []
#             for spl_token in special_tokens:
#                 chunk = chunk.split(spl_token)
#                 # chunk = [c.strip() for c in chunk] # strip may be not needed
#                 new_chunks.extend(chunk)
#             text_chunk = new_chunks
        
#         # using regex implementation
#         # escaped = [re.escape(tok) for tok in special_tokens] # escaped, because we can be special regex characters such as . which will lead to bad matching 
#         # split_pat = f"{"|".join(escaped)}"
#         # text_chunk = re.split(split_pat, text)

        
#         # create word count for each chunk 
#         for chunk in text_chunk:
#             if chunk in special_tokens:
#                 continue
#             for match in re.finditer(PAT, chunk):
#                 self.word_count[tuple([bytes([b]) for b in match.group(0).encode("utf-8")])] += 1
#         # print(self.word_count)
#         # merge characters till we reach the desired vocab size 
#         iter_nums = vocab_size - len(self.vocab)
#         for _ in range(iter_nums):
#             current_pair_count = Counter()
#             # calculate pair stats
#             for word, count in self.word_count.items():
#                 for w1, w2 in pairwise(word):
#                     current_pair_count[(w1, w2)] += count
#             # break
#             if not current_pair_count:
#                 continue

#             # get max_count and merge this pair
#             pair, count = max(current_pair_count.items(), key = lambda x: (x[1], x[0]))
#             self.merges.append(pair)
#             new_token = pair[0] + pair[1]
            
#             if new_token in self.token_to_id: # seen this token earlier
#                 continue
            
#             curr_idx = len(self.vocab)
#             self.vocab[curr_idx] = new_token
#             self.token_to_id[new_token] = curr_idx


#             new_word_count = Counter()
#             for word, count in self.word_count.items():
#                 new_word = []
#                 i = 0
#                 while i < len(word):
#                     if i < len(word)-1 and word[i] == pair[0] and word[i+1] == pair[1]:
#                         new_word.append(new_token)
#                         i += 2
#                     else:
#                         new_word.append(word[i])
#                         i += 1

#                 new_word_count[tuple(new_word)] += count
#             self.word_count = new_word_count
#         return self.vocab, self.merges

class BPEtokenizer():
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.merges = []

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


if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    vocab, merges = tokenizer.train("./data/test_data.txt", vocab_size=260, special_tokens=["<|endoftext|>"])
    print(vocab, "\n\n")
    print(merges, "\n\n")
