import os
from typing import BinaryIO
from multiprocessing import Pool,cpu_count
import regex as re
from collections import defaultdict,Counter

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

def merge_token(token:tuple[bytes,...],freq,pair_to_merge:tuple[bytes,bytes],new_symbol:tuple[bytes],iter_counter:defaultdict[int],pair_to_words:defaultdict[bytes,list],orig_token)->tuple[bytes,...]:
    output=[]
    i=0
    while i<len(token):
        if i<len(token)-1 and ((token[i],)+(token[i+1],))==pair_to_merge:
            output.append(b"".join(new_symbol))
            iter_counter[new_symbol]-=freq
            if i>0:
                iter_counter[(token[i-1],)+(token[i],)]-=freq
                if iter_counter[(token[i-1],)+(token[i],)]==0:
                    iter_counter.pop((token[i-1],)+(token[i],))
                iter_counter[(token[i-1],)+(b"".join(new_symbol),)]+=freq
                pair_to_words[token[i-1]+b"".join(new_symbol)].append(orig_token)
                pair_to_words[token[i-1]+new_symbol[0]].remove(orig_token)
                # pair_to_words[token[i-1]+new_symbol].append(token)
            if i<len(token)-2:
                iter_counter[(token[i+1],)+(token[i+2],)]-=freq
                if iter_counter[(token[i+1],)+(token[i+2],)]==0:
                    iter_counter.pop((token[i+1],)+(token[i+2],))
                iter_counter[(b"".join(new_symbol),)+(token[i+2],)]+=freq
                pair_to_words[b"".join(new_symbol)+token[i+2]].append(orig_token)
                pair_to_words[new_symbol[1]+token[i+2]].remove(orig_token)
                # pair_to_words[new_symbol+token[i+2]].append(token)
            i+=2
        else:
            output.append(token[i])
            i+=1
    new_token=tuple(output)
    # new_symbol=b"".join(new_symbol)
    # for i in range(len(new_token)):
    #     if new_token[i]==new_symbol:
    #         if i>0:
    #             pair_to_words[token[i-1]+new_symbol].append(orig_token)
    #         if i<len(token)-2:
    #             pair_to_words[new_symbol+token[i+2]].append(orig_token)
    return new_token
        

def train_bpe(tokens: list[str],spetial_tokens:list[str],vocab_size:int)-> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab_counter=Counter()
    for token in tokens:
        if token not in spetial_tokens:
            token_encoded=token.encode('utf-8')  # Ensure token is encoded
            vocab_counter[token_encoded]+=1
    num_iteration=vocab_size-len(spetial_tokens)-256
    vocab={i:bytes([i]) for i in range(256)}
    iter_counter=defaultdict(int)
    iter_base = defaultdict(int)
    for voc, freq in vocab_counter.items():
        key = tuple([bytes([c]) for c in voc])
        iter_base[key] = freq
    pair_to_words=defaultdict(list)
    merges=[]
    orig_to_cur={}
    for voc,freq in iter_base.items():
        for i in range(len(voc)-1):
            iter_counter[(voc[i],)+(voc[i+1],)]+=freq
            pair_to_words[voc[i]+voc[i+1]].append(voc)
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
        for orig_word in pair_to_words[new_vocab]:
            word=orig_to_cur[orig_word]
            freq=iter_base[word]
            merged_word=merge_token(word,freq,most_common[0],new_vocab_tuple,iter_counter,pair_to_words,orig_word)
            
            iter_base[merged_word]+=iter_base[word]
            iter_base.pop(word)
            orig_to_cur[orig_word]=merged_word
            if iter_counter[most_common[0]]==0:
                iter_counter.pop(most_common[0])
        
        pair_to_words.pop(b"".join(most_common[0]))
    return vocab, merges
        
            
    


if __name__ == '__main__':

    # tokens=parallel_pretokenize_file(
    #     file_path='/data/home/zhangyichi/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt',
    #     num_chunks=16,
    #     special_tokens=['<|endoftext|>'],
    #     regex_pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # )
    
    # print(tokens[:100])
    
    
    vocab,merges=train_bpe(["low","how","how","are"],[],1000)


    print(vocab)
    print(merges)