'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import re
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader as GraphDataLoader
from data_util import LMGNNDataset
from GPT2.model import LMGNN

def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(args.text)
    context_tokens = enc.encode(args.text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

def train(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=False)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = LMGNN(config)
    model = load_weight(model, state_dict)
    model.to(device)
    # Freeze the LM
    for param in model.parameters():
        param.requires_grad = False
    for param in model.gnn.parameters():
        param.requires_grad = True
    model.train()

    # sent = "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? A) ignore B) enforce C) authoritarian D) yell at E) avoid"
    # context_tokens = enc.encode(sent)
    # answer = 'B'
    # answer_tokens = enc.encode(answer)
    # context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    # for i in range(100):
    #     optimizer.zero_grad()
    #     logits, past = model(context)
    #     logits = logits[:, -1, :].contiguous()
    #     answer = torch.tensor(answer_tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    #     loss = criterion(logits.view(-1, logits.size(-1)), answer.view(-1))
    #     loss.backward()
    #     optimizer.step()

    #     print(loss.item())

    
    batch_size = args.batch_size
    learning_rate = 0.0001
    epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_path = 'data/csqa/train_sents.jsonl'
    dataset = LMGNNDataset(train_path)
    # dataset = torch.utils.data.Subset(dataset, range(50))
    
    sent_graph_path = 'data/csqa/sent_graphs.pt'
    sent_graphs = torch.load(sent_graph_path)
    # sent_graphs_dataset = GraphDataset(sent_graph_path)

    def collate_fn(batch):
        idx, sentences, answers, token_to_node_mappings = zip(*batch)
        tokenized_inputs = []
        for sent in sentences:
            tokenized_sent = []
            for word in re.findall(r'\w+|[^\w\s]', sent):
                tokens = enc.encode(word + ' ')
                tokenized_sent = tokenized_sent + tokens
            tokenized_inputs.append(tokenized_sent)

        # tokenized_inputs = [tokenizer.encode(sent, bos=True, eos=False) for sent in sentences]

        token_to_node_mappings = [torch.tensor(mapping, dtype=torch.long) for mapping in token_to_node_mappings]
        token_to_node_mappings_padded = pad_sequence(token_to_node_mappings, batch_first=True, padding_value=-1)

        return idx, tokenized_inputs, answers, token_to_node_mappings_padded

    data_loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)


    for epoch in range(epochs):
        total_loss = 0
        for i, (idx, sent_tokens, answer, token_to_node) in enumerate(data_loader):
            graphs_loader = GraphDataLoader(
                [sent_graphs[i] for i in idx], 
                batch_size, shuffle=False, 
            )
            graphs = next(iter(graphs_loader)).to(device)
            optimizer.zero_grad()

            context = torch.tensor(sent_tokens[0], device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            context_with_answer = torch.cat((context, torch.tensor(enc.encode(answer[0]), device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)), dim=1)
            # out = sample_sequence(
            #     model=model, length=args.length,
            #     context=sent_tokens[0]  if not  args.unconditional else None,
            #     start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            #     batch_size=args.batch_size,
            #     temperature=args.temperature, top_k=args.top_k, device=device
            # )
            # generated = 0
            # out = out[:, len(sent_tokens):].tolist()
            # generated += 1
            # text = enc.decode(out[0])
            # if args.quiet is False:
            #     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            # print(text)
            logits1, _ = model(context, token_to_node_index=token_to_node, graphs=graphs)
            logits1 = logits1[:, -1, :].contiguous()
            answer = torch.tensor(enc.encode(answer[0]), device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            loss_1 = criterion(logits1.view(-1, logits1.size(-1)), answer.view(-1))

            token_to_node = torch.cat((token_to_node, torch.tensor([-1], dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)), dim=1)
            logits2, _ = model(context_with_answer, token_to_node_index=token_to_node, graphs=graphs)
            logits2 = logits2[:, -1, :].contiguous()
            eos = torch.tensor(enc.encoder['<|endoftext|>'], device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            loss_2 = criterion(logits2.view(-1, logits2.size(-1)), eos.view(-1))
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch: {}'.format(epoch))
        total_loss /= len(data_loader)
    torch.save(model, 'model.pth')


    # sent_graph_path = 'data/csqa/sent_graphs.pt'
    # sent_graphs = torch.load(sent_graph_path, map_location=torch.device('cuda'))
    



if __name__ == '__main__':
    if os.path.exists('pretrained_checkpoints/gpt2-pytorch_model.bin'):
        state_dict = torch.load('pretrained_checkpoints/gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        # text_generator(state_dict)
        train(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
