
import os
from typing import BinaryIO
from multiprocessing import Pool,cpu_count
import regex as re
from collections import defaultdict,Counter
import json



import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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

def split_on_special_tokens(text:str,special_tokens:list[str])->list[str]:
    pattern="|".join([re.escape(st)for st in special_tokens])
    return re.split(f'{pattern}',text)

def pretokenize_chunk(text:str,special_tokens:list[str],regex_pattern:str)->list[str]:
    chunks=split_on_special_tokens(text,special_tokens)
    tokens=[]
    for chunk in chunks:
        tokens.extend([pre_token.group() for pre_token in re.finditer(regex_pattern,chunk)])
    return tokens

def process_chunk_file(args):
    file_path,start,end,special_tokens,regex_pattern=args
    with open(file_path,'rb') as f:
        f.seek(start)
        chunk=f.read(end-start).decode('utf-8',errors='ignore')
        return pretokenize_chunk(chunk,special_tokens,regex_pattern)

def parallel_pretokenize_file(file_path:str,num_chunks:int,special_tokens:list[str],regex_pattern:str)->list[str]:
    with open(file_path,'rb') as f:
        boundaries=find_chunk_boundaries(f,num_chunks,special_tokens[0].encode('utf-8'))
        chunk_args=[(file_path,start,end,special_tokens,regex_pattern) for start,end in zip(boundaries[:-1],boundaries[1:])]

        with Pool(min(num_chunks,cpu_count())) as pool:
            results=pool.map(process_chunk_file,chunk_args)
    return [tok for sublist in results for tok in sublist]

def merge_token(token:tuple[bytes,...],freq,new_symbol:tuple[bytes,bytes],iter_counter:defaultdict[int],pair_to_words:defaultdict[bytes,list],orig_token)->tuple[bytes,...]:


    for i in range(len(token)-1):
        a,b=token[i],token[i+1]
        iter_counter[(a,b)]-=freq
        if iter_counter[(a,b)]==0:
            iter_counter.pop((a,b))
        pair_to_words[a+b].discard(orig_token)
        
    output=[]
    i=0
    while i < len(token):
        if i<len(token)-1 and ((token[i],token[i+1]))==new_symbol:
            output.append(b"".join(new_symbol))
            i+=2
        else:
            output.append(token[i])
            i+=1

    new_token=tuple(output)
    for i in range(len(new_token)-1):
        a,b=new_token[i],new_token[i+1]
        iter_counter[(a,b)]+=freq
        pair_to_words[a+b].add(orig_token)
        
    return new_token
            
    
    
        

def train_bpe(tokens: list[str],spetial_tokens:list[str],vocab_size:int)-> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab_counter=Counter()
    for token in tokens:
        if token not in spetial_tokens:
            token_encoded=token.encode('utf-8')  # Ensure token is encoded
            vocab_counter[token_encoded]+=1
    num_iteration=vocab_size-len(spetial_tokens)-256
    vocab = {}
    # 先加 special tokens
    for i, token in enumerate(spetial_tokens):
        vocab[i] = token.encode("utf-8")

    # 然后加 byte-level vocab，从 len(special_tokens) 开始
    offset = len(spetial_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])
    iter_counter=defaultdict(int)
    iter_base = defaultdict(int)
    for voc, freq in vocab_counter.items():
        key = tuple([bytes([c]) for c in voc])
        iter_base[key] = freq
    pair_to_words=defaultdict(set)
    merges=[]
    orig_to_cur={}
    for voc,freq in iter_base.items():
        for i in range(len(voc)-1):
            iter_counter[(voc[i],)+(voc[i+1],)]+=freq
            pair_to_words[voc[i]+voc[i+1]].add(voc)
        orig_to_cur.update({voc:voc})    
    for i in range(num_iteration):
        if len(iter_counter)==0:
            break
        most_common=max(iter_counter.items(),key=lambda x:(x[1],x[0]))
        new_index=len(vocab)
        new_vocab_tuple=most_common[0]
        merges.append((new_vocab_tuple[0],new_vocab_tuple[1]))
        new_vocab=b"".join(new_vocab_tuple)
        vocab.update({new_index:new_vocab})
        for orig_word in list(pair_to_words[new_vocab]):
            word=orig_to_cur[orig_word]
            freq=iter_base[word]
            merged_word=merge_token(word,freq,new_vocab_tuple,iter_counter,pair_to_words,orig_word)
            
            iter_base[merged_word]+=iter_base[word]
            iter_base.pop(word)
            orig_to_cur[orig_word]=merged_word
            if iter_counter[most_common[0]]==0:
                iter_counter.pop(most_common[0])
        
        pair_to_words.pop(b"".join(most_common[0]))
    return vocab, merges



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    # raise NotImplementedError
    regex_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    tokens=parallel_pretokenize_file(
        file_path=input_path,
        num_chunks=16,
        special_tokens=special_tokens,
        regex_pattern=regex_pattern
    )
    
    return train_bpe(tokens,special_tokens,vocab_size)

def save_to_file(vocab,merges):
    gpt2_byte_encoder=gpt2_bytes_to_unicode()
    with open("owt_merges.txt","w") as f:
        merges=[("".join([gpt2_byte_encoder[token] for token in merge_token_1]),
                "".join([gpt2_byte_encoder[token] for token in merge_token_2])) for merge_token_1, merge_token_2 in merges]  
        f.writelines([f"{merge_token_1} {merge_token_2}\n" for merge_token_1, merge_token_2 in merges]) 
    with open("owt_vocab.json","w") as f:
        vocab = {
            "".join([gpt2_byte_encoder[token] for token in gpt2_vocab_item]):gpt2_vocab_index
            for  gpt2_vocab_index,gpt2_vocab_item in vocab.items()
        } 
        json.dump(vocab,f,ensure_ascii=False,indent=4)


if __name__ == '__main__':

    # tokens=parallel_pretokenize_file(
    #     file_path='/data/home/zhangyichi/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt',
    #     num_chunks=16,
    #     special_tokens=['<|endoftext|>'],
    #     regex_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # )
    
    # print(tokens[:100])
    
    
    vocab,merges=run_train_bpe(
        input_path='/workspace/home/luotianwei/cs336/assignment1-basics/data/owt_train.txt',
        vocab_size=10000,
        special_tokens=['<|endoftext|>'],
    )
    
    save_to_file(vocab,merges)

    # print(vocab)
    # print(merges)
    