import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import tqdm
import re
from pathlib import Path

from llama import Llama
from typing import List
from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
from data_util import LMGNNDataset

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

ckpt_dir = "models/llama-2-7b/"
tokenizer_path = 'tokenizer.model'
max_seq_len: int = 128
max_gen_len: int = 64
top_p: float = 0.9
temperature: float = 0.6
max_batch_size: int = 4

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

prefix = 'Here is a multiple choice question: '
suffix = ' The answer to this question is:'

def collate_fn(batch):
    _, sentences, answers, _ = zip(*batch)

    return sentences, answers

data_path = 'data/csqa/train_sents.jsonl'
dataset = LMGNNDataset(data_path)

num_samples = 20
sample_dataset = Subset(dataset, range(num_samples))

data_loader = DataLoader(sample_dataset, max_batch_size, shuffle=False, generator=torch.Generator(device='cuda'), collate_fn=collate_fn) 

for sentences, answers in data_loader:
    sentences = [prefix + sentence + suffix for sentence in sentences]
    results = generator.text_completion(
        sentences,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result, answer in zip(sentences, results, answers):
        print(prompt),
        print(f"> {result['generation']}")
        print(f"> True answer: {answer}")
        print("\n==================================\n")

# # Model Preparation
# ckpt_dir = "models/llama-2-7b/"
# tokenizer_path = 'tokenizer.model'
# with open(Path(ckpt_dir) / "params.json", "r") as f:
#             params = json.loads(f.read())

# checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
# ckpt_path = checkpoints[0]
# checkpoint = torch.load(ckpt_path)

# tokenizer = Tokenizer(model_path=tokenizer_path)

# model_args: ModelArgs = ModelArgs(
#             max_seq_len=512,
#             max_batch_size=8,
#             **params,
#         )
# model_args.vocab_size = tokenizer.n_words

# device = torch.device("cuda")

# if not torch.distributed.is_initialized():
#     torch.distributed.init_process_group("nccl")
# model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
# initialize_model_parallel(model_parallel_size)

# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# torch.cuda.set_device(local_rank)

# torch.manual_seed(1)

# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# model = Transformer(model_args)

# model.load_state_dict(checkpoint, strict=False)

# batch_size = 8

