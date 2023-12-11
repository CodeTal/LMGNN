import networkx as nx
import pickle
import torch
from tqdm import tqdm
import json
import re
import os

import torch_geometric
from torch_geometric.data import Data
from sentencepiece import SentencePieceProcessor

import nltk

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

def construct_graph_pyg(cpnet_csv_path, cpnet_vocab_path, output_path, prune=True):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    concept2id = {}
    id2concept = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

    x = torch.tensor([i for i in range(len(concept2id))], dtype=torch.float)
    edge_index = [[],[]]
    edge_attr = []
    edge_weight = []
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???
            if (subj, obj, rel) not in attrs:
                edge_index[0].append(subj)
                edge_index[1].append(obj)
                edge_attr.append(rel)
                edge_weight.append(weight)
                attrs.add((subj, obj, rel))
                edge_index[0].append(obj)
                edge_index[1].append(subj)
                edge_attr.append(rel + len(relation2id))
                edge_weight.append(weight)
                attrs.add((obj, subj, rel + len(relation2id)))
                # graph.add_edge(subj, obj, rel=rel, weight=weight)
                # attrs.add((subj, obj, rel))
                # graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                # attrs.add((obj, subj, rel + len(relation2id)))
    edge_index = torch.tensor(edge_index, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # graph.edge_weight = edge_weight
    torch.save(graph, output_path)
    print(graph.edge_attr)
    graph = torch.load(output_path)
    print(graph.edge_attr)
    # nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    with open (cpnet_graph_path, 'rb') as f:
        cpnet = pickle.load(f)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

def create_LMGNN_data(grounded_path, qafile_path, cpnet_graph_path, cpnet_vocab_path, output_sents_path, output_graphs_path, tokenizer_path):
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    load_resources(cpnet_vocab_path)
    load_cpnet(cpnet_graph_path)

    sp = SentencePieceProcessor(model_file=tokenizer_path)

    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(qafile_path, 'r', encoding='utf-8') as fin_qa, open(output_sents_path, 'w', encoding='utf-8') as fout:
        lines_qa  = fin_qa.readlines()
        lines_ground = fin_ground.readlines()
        assert len(lines_ground) % len(lines_qa) == 0
        n_choices = len(lines_ground) // len(lines_qa)
        sent_graphs = []

        for i, line_qa in enumerate(lines_qa):
            dic_qa= json.loads(line_qa)
            model_input = dic_qa['question']['stem'] + ' '
            choices = dic_qa['question']['choices']
            answer = ''
            for choice in choices:
                model_input += choice['label'] + ') ' + choice['text'] + ' '
                if choice['label'] == dic_qa['answerKey']:
                    answer = choice['text']
            model_input = model_input[:-1]
            model_output = dic_qa['answerKey']

            # build the set for concepts in the question and answer
            qids = set()
            a_ids = set()
            for j in range(i * n_choices, (i + 1) * n_choices):
                line_ground = lines_ground[j]
                dic_ground = json.loads(line_ground)
                qids |= set(concept2id[c] for c in dic_ground['qc'])
                a_ids |= set(concept2id[c] for c in dic_ground['ac'])
            q_ids = qids - a_ids
            qa_nodes = set(q_ids) | set(a_ids)
            extra_nodes = set()
            for qid in qa_nodes:
                for aid in qa_nodes:
                    if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                        extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
            extra_nodes = extra_nodes - qa_nodes

            x = list(q_ids) + list(a_ids) + list(extra_nodes)
            edge_index = [[],[]]
            for s in x:
                for t in x:
                    if s != t:
                        if cpnet_simple.has_edge(s, t):
                            edge_index[0].append(x.index(s))
                            edge_index[1].append(x.index(t))
                            edge_index[0].append(x.index(t))
                            edge_index[1].append(x.index(s))
            qa_concept_ids = [item for item in x]
            x = torch.tensor([[item] for item in x], dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            sent_graph = Data(x=x, edge_index=edge_index)
            sent_graphs.append(sent_graph)

            # Assign each token with an index of the corresponding node in the graph, -1 for no matching nodes
            # 
            token_to_node = []
            for word in re.findall(r'\w+|[^\w\s]', model_input):
                tokens = sp.encode(word)
                if word in concept2id and concept2id[word] in qa_concept_ids:
                    token_to_node += [qa_concept_ids.index(concept2id[word])] * len(tokens)
                else:
                    token_to_node += [-1] * len(tokens)
            
            training_dic = {'sent': model_input, 'answer': model_output, 'token_to_node': token_to_node}
            fout.write(json.dumps(training_dic) + '\n')
        torch.save(sent_graphs, output_graphs_path)
            # print(model_input)
            # print(sp.encode(model_input))
            # print(token_to_node)
            # print()
            # show = ''
            # for i in range(len(token_to_node)):
            #     if token_to_node[i] != -1:
            #         show += id2concept[qa_concept_ids[token_to_node[i]]] + ' '
            # print(show)
                    
                    
                    



    # node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    # node_mask[graph.x] = True
    # return node_mask

if __name__ == '__main__':

    create_LMGNN_data(
        './data/csqa/grounded/train.grounded.jsonl',
        './data/csqa/train_rand_split.jsonl',
        './data/cpnet/conceptnet.en.unpruned.graph',
        './data/cpnet/concept.txt',
        './data/csqa/train_sents.jsonl',
        './data/csqa/sent_graphs.pt',
        './tokenizer.model'
    )

# with open('./data/cpnet/conceptnet.en.unpruned.graph', 'rb') as f:
#     G = pickle.load(f)
#     res = range(10)
#     k=G.subgraph(res)
#     G = pyg.utils.convert.from_networkx(k)
#     print(G)
    # pos = nx.spring_layout(k)
    # pl.figure()
    # nx.draw(k, pos=pos)
    # pl.show()
    # G = pyg.utils.convert.from_networkx(nx_G)
