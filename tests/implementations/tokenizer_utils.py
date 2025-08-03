import argparse
import numpy as np
from pathlib import Path
from tokenizer import Tokenizer  # 替换为你自己的路径


def tokenize_chunked_file(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
    chunk_size: int = 1 << 20,  # 每次读取 1MB
    separator: str = "<|endoftext|>"
) -> None:
    tokens = []
    leftover = ""

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            text = leftover + chunk
            last_sep = text.rfind(separator)

            if last_sep == -1:
                # 没有分隔符，先编码整个 chunk
                encode_text = text
                leftover = ""
            else:
                # 截断在最近一个 separator
                encode_text = text[:last_sep + len(separator)]
                leftover = text[last_sep + len(separator):]

            tokens.extend(tokenizer.encode(encode_text))

        # 最后一段也处理
        if leftover.strip():
            tokens.extend(tokenizer.encode(leftover))

    tokens_np = np.array(tokens, dtype=np.uint16)
    np.save(output_path, tokens_np)
    print(f"✅ Saved {len(tokens_np)} tokens to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", help="Path to input .txt file")
    parser.add_argument("--output", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/data/TinyStoriesV2_GPT4-train.npy", help="Path to output .npy file")
    parser.add_argument("--vocab_path", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/training_result/tinystory_vocab.json", help="Path to vocab .npy file")
    parser.add_argument("--merges_path", type=str, default="/workspace/home/luotianwei/cs336/assignment1-basics/training_result/tinystory_merges.txt", help="Path to merges .npy file")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens to preserve")
    parser.add_argument("--chunk_size", type=int, default=1 << 20, help="Chunk size in bytes (default: 1MB)")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)

    tokenize_chunked_file(
        input_path=args.input,
        output_path=args.output,
        tokenizer=tokenizer,
        chunk_size=args.chunk_size,
        separator="<|endoftext|>"
    )

if __name__ == "__main__":
    main()