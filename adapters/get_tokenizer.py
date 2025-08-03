from typing import Iterable
import regex as re
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

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> "Tokenizer":
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


class Tokenizer():
    def __init__(self,vocab,merges,special_tokens=None)->"Tokenizer":
        self.vocab=vocab
        self.bytes_to_id={b:a for a,b in vocab.items()}
        self.merges=merges
        self.special_tokens=special_tokens if special_tokens is not None else []
        self.regex_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
    def split_on_special_tokens(self,text:str,special_tokens:list[str])->list[str]:
        pattern=f"({"|".join([re.escape(st)for st in special_tokens])})"
        return re.split(f'{pattern}',text)
    
    def pretokenize(self,text:str)->list[bytes]:
        chunks=self.split_on_special_tokens(text,self.special_tokens)
        tokens=[]
        for chunk in chunks:
            if chunk not in self.special_tokens:
                tokens.append([pre_token.group() for pre_token in re.finditer(self.regex_pattern,chunk)])
            else:
                tokens.append(chunk)
        return tokens
    
    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None)->"Tokenizer":
        vocab, merges = cls.transform_from_file(vocab_filepath, merges_filepath, special_tokens)
        return cls(vocab, merges, special_tokens)
    
    @classmethod
    def transform_from_file(cls,vocab_filepath:str,merges_filepath:str,special_tokens=None)->"Tokenizer":
        gpt2_byte_decoder={v:k for  k,v in gpt2_bytes_to_unicode().items()}
        with open(merges_filepath) as f:
            merges=[tuple(line.rstrip().split(" ")) for line in f]
            merges=[(bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                     bytes([gpt2_byte_decoder[token] for token in merge_token_2])) for merge_token_1, merge_token_2 in merges]   
        with open(vocab_filepath) as f:
            vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in vocab.items()
            }       
        return vocab,merges 
            
    def encode_pure(self,tokens:list[str])->list[int]:
        code=[]
        for token in tokens:
            token=tuple(bytes([byte]) for byte in token.encode('utf-8'))
            for merge in self.merges:
                a,b=merge
                i=0
                new_token=[]
                while i < len(token):
                    if i<len(token)-1 and token[i] == a and token[i + 1] == b:
                        new_token.append(a + b)
                        i+=2
                    else:
                        new_token.append(token[i])
                        i += 1
                token=new_token
            code.extend([self.bytes_to_id[st] for st in token ])
        return code
    
    def encode(self,text:str)-> list[int]:
        code=[]
        tokens=self.pretokenize(text)
        for token in tokens:
            if type(token)==str:
                code.append(self.bytes_to_id[token.encode('utf-8')])
            else:
                code.extend(self.encode_pure(token))
        return code
                
    
    def encode_iterable(self,iterable:Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self,ids:list[int])-> str:
        raw_utf8=b"".join([self.vocab[id] for id in ids])
        return raw_utf8.decode('utf-8')
    
if __name__ == "__main__":
    # Example usage
    vocab = {0: b"<|endoftext|>", 1: b"Hello", 2: b"World"}
    merges = [(b"Hello", b"World")]
    special_tokens = ["<|endoftext|>"]

    tokenizer=Tokenizer.from_files("/workspace/home/luotianwei/cs336/assignment1-basics/self_vocab.json","/workspace/home/luotianwei/cs336/assignment1-basics/self_merges.txt",['<|endoftext|>'])
    
    print(tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"))
    print(tokenizer.decode(tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")))
    print('aaa')
    
