from .model import TransformerLM
from .tokenizer import Tokenizer

import torch
import random
import numpy as np

def inference(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_len: int = 256) -> str:
    """
    Perform inference using the TransformerLM model.

    Args:
        model (TransformerLM): The pre-trained Transformer language model.
        tokenizer (Tokenizer): The tokenizer to encode and decode text.
        prompt (str): The input text prompt for the model.
        max_length (int): The maximum length of the generated sequence.

    Returns:
        str: The generated text sequence.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize the input prompt
        input_ids = tokenizer.encode(prompt)  # Add batch dimension
        input_ids=torch.tensor(input_ids, dtype=torch.long,device="cuda") # Add batch dimension

        # Generate output sequence
        output_ids = model.generate(input_ids, eos_token_id=0,max_len=max_len,temperature=0.7,top_p=0.85)

        # Decode the output sequence back to text
        generated_text = tokenizer.decode(output_ids)

    return generated_text


def set_seed(seed: int = 42):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # Example usage
    # set_seed(42)
    model = TransformerLM(vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=4, d_ff=1344, rope_theta=10000.0).to('cuda')  # Initialize your model with the appropriate parameters
    tokenizer = Tokenizer.from_files('/workspace/home/luotianwei/cs336/assignment1-basics/trained_tokenizer/tinystory_vocab.json','/workspace/home/luotianwei/cs336/assignment1-basics/trained_tokenizer/tinystory_merges.txt',['<|endoftext|>'])  # Initialize your tokenizer with the appropriate vocab size
    state_dict= torch.load('/workspace/home/luotianwei/cs336/assignment1-basics/checkpoints/stantard_setting/checkpoint_38000.pt')
    model_state_dict=state_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    prompt = "How to right code efficiently? This is a good question."
    generated_text = inference(model, tokenizer, prompt, max_len=256)
    print(generated_text)
