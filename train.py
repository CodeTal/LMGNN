import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import tqdm
import re
from pathlib import Path

from llama.model import LMGNN, ModelArgs
from llama.tokenizer import Tokenizer
from data_util import LMGNNDataset

from torch_geometric.loader import DataLoader as GraphDataLoader


from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

# Model Preparation
ckpt_dir = "models/llama-2-7b/"
tokenizer_path = 'tokenizer.model'
with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
ckpt_path = checkpoints[0]
checkpoint = torch.load(ckpt_path)

tokenizer = Tokenizer(model_path=tokenizer_path)

model_args: ModelArgs = ModelArgs(
            max_seq_len=512,
            max_batch_size=8,
            **params,
        )
model_args.vocab_size = tokenizer.n_words

device = torch.device("cuda")

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
initialize_model_parallel(model_parallel_size)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

torch.manual_seed(1)

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = LMGNN(model_args, in_dim=1, hidden_dim=200, out_dim=model_args.dim, heads=4)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model.gnn.float()

model.load_LM_state_dict(checkpoint)

# Freeze the LM
for param in model.parameters():
    param.requires_grad = False
for param in model.layers[-1].parameters():
    param.requires_grad = True

# Dataset Preparation
batch_size = 1

train_path = 'data/csqa/train_sents.jsonl'
dataset = LMGNNDataset(train_path)

sent_graph_path = 'data/csqa/sent_graphs.pt'
sent_graphs = torch.load(sent_graph_path, map_location=torch.device('cuda'))
# sent_graphs_dataset = GraphDataset(sent_graph_path)

def collate_fn(batch):
    idx, sentences, answers, token_to_node_mappings = zip(*batch)
    tokenized_inputs = []
    for sent in sentences:
        tokenized_sent = [tokenizer.bos_id]
        for word in re.findall(r'\w+|[^\w\s]', sent):
            tokens = tokenizer.encode(word, bos=False, eos=False)
            tokenized_sent = tokenized_sent + tokens
        tokenized_inputs.append(tokenized_sent)

    # tokenized_inputs = [tokenizer.encode(sent, bos=True, eos=False) for sent in sentences]
    
    token_to_node_mappings = [torch.tensor([tokenizer.bos_id] + mapping, dtype=torch.long) for mapping in token_to_node_mappings]
    token_to_node_mappings_padded = pad_sequence(token_to_node_mappings, batch_first=True, padding_value=-1)

    return idx, tokenized_inputs, answers, token_to_node_mappings_padded

data_loader = DataLoader(dataset, batch_size, shuffle=False, generator=torch.Generator(device='cuda'), collate_fn=collate_fn)

# Training
learning_rate = 0.00001
num_epochs = 10

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

temperature = 0.6

# graphs_loader = GraphDataLoader(
#             [sent_graphs[i] for i in range(50)], 
#             batch_size, shuffle=False, 
#             generator=torch.Generator(device='cuda')
#             )
# graphs = next(iter(graphs_loader))
# graphs = torch.tensor(graphs.to_list())




# train_sent = 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? A) ignore B) enforce C) authoritarian D) yell at E) avoid'
# train_sent = tokenizer.encode(train_sent, bos=True, eos=False)
# train_sent = torch.tensor(train_sent, dtype=torch.long, device="cuda").unsqueeze(0)

# print(model(train_sent, _, 0, graphs))
# print('AA')





for epoch in range(num_epochs):
    total_loss = 0
    for i, (idx, sent_tokens, answer, token_to_node) in enumerate(data_loader):
        graphs_loader = GraphDataLoader(
            [sent_graphs[i] for i in idx], 
            batch_size, shuffle=False, 
            generator=torch.Generator(device='cuda')
            )
        graphs = next(iter(graphs_loader))

        min_prompt_len = min(len(t) for t in sent_tokens)
        max_prompt_len = max(len(t) for t in sent_tokens)
        max_gen_len = 1
        # max_seq_len = 512
        # total_len = min(max_seq_len, max_gen_len + max_prompt_len)
        total_len = min_prompt_len + 1

        pad_id = tokenizer.pad_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(sent_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        eos_reached = torch.tensor([False] * batch_size, device="cuda")

        prev_pos = 0
        input_text_mask = tokens != pad_id

        assert min_prompt_len + 1 == total_len
        optimizer.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
            logits = model(tokens[:, prev_pos:min_prompt_len], token_to_node, prev_pos, graphs)
            tokenized_ans = torch.tensor(tokenizer.encode(answer[0], bos=False, eos=False), dtype=torch.long, device="cuda")
            # tokenized_ans = tokenized_ans.detach()
            # eos_reached = eos_reached.detach()
            
            # loss = loss_function(logits[-1:, :], tokenized_ans.view(-1)).to(device)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            # print(probs.shape)
            # print(probs[0, :10])
            # print(logits[:, -1, :10])
            loss = loss_function(probs, tokenized_ans.view(-1)).to(device)
            print(probs[:30])
            print(loss)
            # from torchviz import make_dot
            # make_dot(loss).render("attached", format="pdf")
        # print('AA')
        # print(epoch)
        # print(i)
        # print(sent_tokens)
        # print(loss)
        # print('BB')
        # pr-int('AA')
            print(loss)

            # for name,parameters in model.layers[-1].named_parameters():
            #     print(name,':',parameters)
            #     print(name,':',parameters.grad)

            loss.backward()
            optimizer.step()
            # total_loss = total_loss + loss.item()

        # if min_prompt_len == total_len:
        #     logits = model.forward(tokens, prev_pos)
        #     token_logprobs = -F.cross_entropy(
        #         input=logits.transpose(1, 2),
        #         target=tokens,
        #         reduction="none",
        #         ignore_index=pad_id,
        #     )
        
        # for cur_pos in range(min_prompt_len, total_len):
        #     _ = ''
        #     logits = model.forward(tokens[:, prev_pos:cur_pos], token_to_node, prev_pos, graphs)
        #     next_token = torch.argmax(logits[:, -1], dim=-1)
        #     next_token = next_token.reshape(-1)
        #     next_token = torch.where(
        #         input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        #     )
        #     tokens[:, cur_pos] = next_token
        #     eos_reached |= (~input_text_mask[:, cur_pos]) & (
        #         next_token == tokenizer.eos_id
        #     )
        #     prev_pos = cur_pos
        #     if all(eos_reached):
        #         break

        # out_tokens = []
        # for i, token in enumerate(tokens.tolist()):
        #     start = 0
        #     token = token[start:len(sent_tokens[i]) + max_gen_len]
        #     if tokenizer.eos_id in token:
        #         eos_idx = token.index(tokenizer.eos_id)
        #         token = token[:eos_idx]
        #     out_tokens.append(token)

        # total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        # eos_reached = torch.tensor([False] * batch_size, device="cuda")

        # min_prompt_len = min(len(t) for t in sent_tokens)
        # max_prompt_len = max(len(t) for t in sent_tokens)
        # prev_pos = 0