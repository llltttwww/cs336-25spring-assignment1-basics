import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Int, Float, Bool
from typing import Optional
from torch import Tensor


class Linear(nn.Module):
    def __init__(self,in_features: int,out_features:int,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        factory_kwargs={"device":device,"dtype":dtype}
        
        self.weight=nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        std=2/(in_features+out_features)**0.5
        nn.init.trunc_normal_(self.weight,std=std,a=-3*std,b=3*std)
        
    def forward(self,x:Float[Tensor,"... d_in"])->Float[Tensor,"b d_out"]:
        return einsum(x,self.weight,"... d_in, d_out d_in -> ... d_out")
    


class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))
        std=1
        nn.init.trunc_normal_(self.weight,std=std,a=-3,b=3)
        
    def forward(self,token_ids:Tensor) -> Tensor:
        return self.weight[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps=eps
        factory_kwargs={"device":device,"dtype":dtype}
        self.weight=nn.Parameter(torch.ones((d_model,),**factory_kwargs))
        
    def forward(self,x:Float[Tensor,"... d_model"]) -> Float[Tensor,"... d_model"]:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        x=x/rms
        x*=self.weight
        return x.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        factory_kwargs={"device":device,"dtype":dtype}
        
        self.w1=Linear(d_model,d_ff,**factory_kwargs)
        self.w3=Linear(d_model,d_ff,**factory_kwargs)
        self.w2=Linear(d_ff,d_model,**factory_kwargs)
        
    def forward(self,x:Float[Tensor,"... d_model"]) -> Float[Tensor, "... d_model"]:
        x1=self.w1(x)*torch.sigmoid(self.w1(x))
        x3=self.w3(x)
        xx=einsum(x1,x3,"... d_ff , ... d_ff -> ... d_ff")
        return self.w2(xx)

def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.factory_kwargs={"device":device}
        
        # inv_freq=1.0/(self.theta**(torch.arange(1,self.d_k//2+1,device=device)*2/self.d_k)).float()
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k // 2, device=device).float() * 2 / self.d_k))
        t_freq=torch.arange(0,self.max_seq_len,device=device).float()
        freqs=einsum(inv_freq,t_freq,"d,t->t d")
        
        sin=torch.sin(freqs)
        cos=torch.cos(freqs)
        
        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)
        
    def forward(self,x:Float[Tensor,"... seq_len d_k"], token_positions: Float[Tensor,"... seq_len"]) -> Float[Tensor,"... seq_len d_k"]:
        x_even=x[...,::2]
        x_odd=x[...,1::2]
        if token_positions is None:
            token_positions = torch.arange(0,x.shape[-2],device=x.device).long()
        cos=self.cos[token_positions]
        cos=rearrange(cos,"... t d -> ... 1 t d")
        sin=self.sin[token_positions]
        sin=rearrange(sin,"... t d -> ... 1 t d")
        
        x_even_rotated=x_even*cos-x_odd*sin
        x_odd_rotated=x_even*sin+x_odd*cos
        
        x_rotated=torch.stack((x_even_rotated,x_odd_rotated),dim=-1)
        
        return x_rotated.reshape(x.shape)
    
def softmax(x:Float[Tensor, "..."],dim: Int) -> Float[Tensor,"... d"]:
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_stable=x-x_max
    x_exp=torch.exp(x_stable)
    sum_exp=torch.sum(x_exp,dim=dim,keepdim=True)
    return x_exp/sum_exp

def scaled_dot_product_attention( Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,) -> Float[Tensor, "... queries d_v"]:
    dot_product=einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys")
    scaled_dot_product=dot_product/(Q.shape[-1]**0.5)
    if mask is not None:
        mask=torch.where(mask,0,float("-inf"))
        scaled_dot_product=scaled_dot_product+mask
    attention_weights=softmax(scaled_dot_product,dim=-1)
    attention_output=einsum(attention_weights,V,"... queries keys, ... keys d_v -> ... queries d_v")
    return attention_output
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,rope:Optional[nn.Module]=None, device=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.rope=rope
        self.factory_kwargs={"device":device}
        
        d_in=d_model
        d_k=d_v=d_model//num_heads
        
        self.q_proj: Float[Tensor, " h*d_k d_in"]=Linear(d_in,self.num_heads*d_k,**self.factory_kwargs)
        self.k_proj: Float[Tensor, " h*d_k d_in"]=Linear(d_in,self.num_heads*d_k,**self.factory_kwargs)
        self.v_proj: Float[Tensor, " h*d_v d_in"]=Linear(d_in,self.num_heads*d_v,**self.factory_kwargs)
        self.output_proj: Float[Tensor, " d_model h*d_v"]=Linear(self.num_heads*d_v,d_model,**self.factory_kwargs)
        
    def forward(self,x:Float[Tensor,"... d_in"],token_positions:Optional[Int[Tensor, "... seq_len"]]=None) -> Float[Tensor, "... d_model"]:
        batch_size,seq_len,d_in=x.shape
        d_k=d_v=self.d_model//self.num_heads
        # x=rearrange(x,"batch seq d_in -> batch seq num_head d_in",num_head=self.num_heads)
        # x=rearrange(x,"batch seq num_head d_in -> (batch num_head) seq d_in")
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)
        
        Q=rearrange(Q,"batch seq_len (num_heads d_k) -> (batch num_heads) seq_len d_k",num_heads=self.num_heads)
        K=rearrange(K,"batch seq_len (num_heads d_k) -> (batch num_heads) seq_len d_k",num_heads=self.num_heads)
        V=rearrange(V,"batch seq_len (num_heads d_v) -> (batch num_heads) seq_len d_v",num_heads=self.num_heads)
        
        if self.rope is not None:
            Q=self.rope(Q,token_positions)
            K=self.rope(K,token_positions)
            
        
        def causal_attention_mask(seq_len:int,device=None) -> Bool[Tensor, "query key"]:
            mask=~torch.triu(torch.ones((seq_len,seq_len),dtype=torch.bool,device=device),diagonal=1)
            return mask
        
        attention_mask=causal_attention_mask(seq_len,device=x.device)
        
        attention_output=scaled_dot_product_attention(Q,K,V,attention_mask)
        
        attention_output=rearrange(attention_output,"(batch num_head) seq d_v -> batch seq (num_head d_v)",num_head=self.num_heads)
        
        output=self.output_proj(attention_output)
        
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float,device=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        factory_kwargs={"device":device}
        self.rope=RotaryPositionEmbedding(theta,self.d_model//self.num_heads,max_seq_len,device=device)
        
        
        self.attn=MultiHeadSelfAttention(d_model,num_heads,rope=self.rope,device=device)
        self.ln1= RMSNorm(d_model,device=device)
        self.ln2=RMSNorm(d_model,device=device)
        self.ffn= SwiGLU(d_model,d_ff,device=device)
        
    def forward(self,x:Float[Tensor,"... d_model"],token_positions:Optional[Int[Tensor, "... seq_len"]]=None) -> Float[Tensor,"... d_model"]:
        x_res_1=x
        x=self.ln1(x)
        x=self.attn(x,token_positions)
        x+=x_res_1
        
        x_res_2=x
        x=self.ln2(x)
        x= self.ffn(x)
        x+=x_res_2
        
        return x
    

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int, context_length:int, d_model:int,num_layers:int,num_heads:int,d_ff:int,rope_theta:float):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rope_theta=rope_theta
        
        self.token_embeddings=Embedding(vocab_size,d_model)
        
        self.layers=nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)
        ])
        
        self.ln_final=RMSNorm(d_model)
        self.lm_head=Linear(d_model,vocab_size)
        
    def forward(self,token_ids: Int[Tensor,"batch seq_len"],token_positions:Optional[Int[Tensor, "... seq_len"]]=None) -> Float[Tensor, "batch seq_len d_model"]:
        x=self.token_embeddings(token_ids)
        
        for layer in self.layers:
            x=layer(x,token_positions)
        
        x=self.ln_final(x)
        x=self.lm_head(x)
        
        return x
    
    
    @torch.no_grad()
    def generate(self,prompt:Int[Tensor,"seq_len"],eos_token_id:int,max_len:int=1000,temperature:float=1.0,top_p:float=1.0) -> Int[Tensor," <=context+generated"]:
        # batch generate not supported
        device=prompt.device
        generated=prompt.clone()
        
        for _ in range(max_len):
            seq_len = generated.shape[0]
            token_positions = torch.arange(seq_len, device=device).unsqueeze(0)
            input_ids = generated.unsqueeze(0)

            logits = self(input_ids, token_positions=token_positions)
            next_token_logits = logits[0, -1]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = softmax(next_token_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                mask = cum_probs > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum()
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices[sampled].squeeze(0)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=0)
            if next_token.item() == eos_token_id:
                break
        
        return generated
            
            
            
            
    




if __name__ == "__main__":
    from .tokenizer import Tokenizer
    prompt= "Hello, how are you?"
    tokenizer=Tokenizer.from_files("/workspace/home/luotianwei/cs336/assignment1-basics/training_result/tinystory_vocab.json","/workspace/home/luotianwei/cs336/assignment1-basics/training_result/tinystory_merges.txt")
    input_ids=torch.tensor(tokenizer(prompt)).to("cuda")
    model=TransformerLM(vocab_size=tokenizer.vocab_size,context_length=tokenizer.context_length,d_model=512,num_layers=6,num_heads=8,d_ff=2048,rope_theta=10000.0).to("cuda")
    
    