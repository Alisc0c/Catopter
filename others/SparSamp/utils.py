import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch

def generate_binary_message(length: int, filename: str):
    binary_bits = ''.join(random.choice('01') for _ in range(length))
    with open(filename, 'w') as f:
        f.write(binary_bits)
    print(f"message_bits saved to {filename}.")

def read_binary_message(filename: str) -> str:
    with open(filename, 'r') as f:
        binary_bits = f.read()
    return binary_bits

def get_lower_upper_bound(cumulative_probs, v):
    lower_bound = cumulative_probs[v-1] if v > 0 else torch.tensor(0)
    upper_bound = cumulative_probs[v] if v < len(cumulative_probs)-1 else torch.tensor(1)
    SE = [lower_bound.item(), upper_bound.item()]
    return SE

def func_mrn(k_m, n_m, r):
    result = ((k_m / n_m) + r)
    if result >= 1:
        result = result - 1
    return result

def dec2bin(km, lm):
    bin_str = bin(km)[2:]
    return bin_str.zfill(lm)

def load_model(model_name, device):
    print(f"model_name:{model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype='auto').to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        import warnings
        warnings.warn("Failed to load model locally, load gpt2 directly from Transformers library")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model.eval()
    return model, tokenizer

def limit_past(past):
    if past is None:
        return None
    past = list(past)
    # for i in range(len(past)):
    #     past[i] = list(past[i])
    #     for j in range(len(past[i])):
    #         past[i][j] = past[i][j][:, :, -1022:]

    for layer in past.layers:
        if layer.is_initialized:
            layer.keys = layer.keys[:, :, -1022:]
            layer.values = layer.values[:, :, -1022:]

    past = torch.nn.utils.rnn.PackedSequence(past)
    return past

def get_probs_past(model,
                   prev=None,
                   past=None,
                   device='cuda',
                   top_p=1.0):
    # if past is not None:
    #     past = limit_past(past)
    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values

    logits = model_output.logits[0,-1,:].to(device)
    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    probs = F.softmax(logits, dim=-1)

    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        k = (cum_probs > top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices, past
