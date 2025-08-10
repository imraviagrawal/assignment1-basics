import time , psutil, os
from cs336_basics.tokenizer import BPEtokenizer

def main():
    tokenizer = BPEtokenizer()
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # mem 
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 3)  # GB

    vocab, merges = tokenizer.train(input_path, vocab_size, special_tokens)
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 ** 3)  # GB

    with open("./data/tiny_story_train_vocab.txt", "w", encoding="utf-8") as f:
        for idx, token in vocab.items():
            f.write(f"{idx}\t{token}\n")
    with open("./data/tiny_story_train_merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{pair[0]} {pair[1]}\n")


    print(f"Time used: {(end_time - start_time):.2f} seconds")
    print(f"Memory used: {(mem_after - mem_before):.2f} GB")


if __name__ == "__main__":
    # train bpe
    main()