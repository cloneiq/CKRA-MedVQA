import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from os.path import join
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, BertModel
from torch.nn import MultiheadAttention
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class KG_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(KG_Embedding, self).__init__()
        bert_base = 'pre-model/bert-base-uncased'
        self.base_model = BertModel.from_pretrained(bert_base)
        bert_model = nn.Sequential(*list(self.base_model.children())[0:])
        self.ent_embedding = bert_model[0]
        self.ent_emb, self.b_rel_emb, self.ent_mask, self.b_rel_mask = get_kg_emb("slake_kg")
        self.kg_n_layer = 1
        self.comp_layers = nn.ModuleList([CompLayer('add', int(hidden_size / 2)) for _ in range(self.kg_n_layer)])
        self.rel_embs = nn.ParameterList([torch.cat((self.b_rel_emb, self.b_rel_emb), dim=0) for _ in range(self.kg_n_layer)])
        self.rel_mask = torch.cat((self.b_rel_mask, self.b_rel_mask), dim=0)
        self.rel_w = get_param(hidden_size, hidden_size)
        self.ent_drop = nn.Dropout(0.2)
        self.act = nn.Tanh()
        self.L = nn.Linear(hidden_size, hidden_size)
        self.S = nn.Linear(hidden_size, hidden_size)
        self.mea_func = Measure_F(int(hidden_size / 2), int(hidden_size / 2), [200] * 2, [200] * 2)
        self.kg_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, kg, question_embedding):
        ent_emb = self.ent_emb.long()
        ent_emb = self.ent_embedding(ent_emb)
        common = self.act(self.S(ent_emb))
        private = self.act(self.L(ent_emb))
        rel_emb_list = []

        ent_emb = torch.mean(ent_emb, dim=1)
        ent_emb = self.ent_emb_linear(ent_emb)
        ent_emb = torch.stack([ent_emb] * question_embedding.shape[0], dim=0)
        ent_emb = self.attentionKGselector(ent_emb)

        corr = 0
        for comp_layer, rel_emb in zip(self.comp_layers, self.rel_embs):
            rel_emb = rel_emb.long()
            rel_emb = self.ent_embedding(rel_emb)
            ent_emb = self.ent_drop(ent_emb)
            comp_ent_emb1 = comp_layer(kg, common, rel_emb, self.ent_mask, self.rel_mask, question_embedding)
            comp_ent_emb2 = comp_layer(kg, private, rel_emb, self.ent_mask, self.rel_mask, question_embedding)
            ent_emb = torch.cat((comp_ent_emb1, comp_ent_emb2), dim=-1)
            rel_emb_list.append(rel_emb)
            phi_c, phi_p = self.mea_func(comp_ent_emb1, comp_ent_emb2)
            corr = corr + compute_corr(phi_c, phi_p)

        kg_emb = self.act(self.kg_linear(ent_emb.permute(0, 2, 1)).permute(0, 2, 1))
        batch_size = kg_emb.shape[0]
        masks = torch.ones((batch_size, 1, 1, self.top_k))
        return kg_emb, corr, masks

class PCALayer(nn.Module):
    def __init__(self, num_ent, reduced_dim):
        super(PCALayer, self).__init__()
        self.proj_matrix = nn.Parameter(torch.randn(num_ent, reduced_dim))
        self._orthogonalize()

    def _orthogonalize(self):
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.proj_matrix)
            self.proj_matrix.copy_(Q)

    def forward(self, x):
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_projected = torch.einsum('bnh,nr->brh', x_centered, self.proj_matrix)
        return x_projected

def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param

class CompLayer(nn.Module):
    def __init__(self, comp_op, hidden_size):
        super().__init__()
        self.skip = 1
        self.n_ent = 1968
        self.n_rel = 58
        self.comp_op = comp_op
        assert self.comp_op in ['add', 'mul']
        self.h_dim = hidden_size
        self.hidden_size = hidden_size * 2
        self.act = nn.Tanh()
        self.tok_linear = nn.Linear(self.hidden_size, self.h_dim)
        self.key_linear = nn.Linear(self.hidden_size, self.h_dim)
        self.kg_linear = nn.Linear(self.h_dim, self.h_dim)
        self.comp_linear = nn.Linear(16, 1)
        self.k = int(self.n_ent * 0.1)
        self.scale = (self.h_dim / 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attentionKGselector = AttentionBasedKnowledgeSelector(hidden_size=self.h_dim, target_dim=self.k)

    def forward(self, kg, ent_emb, rel_emb, ent_mask, rel_mask, question_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            kg.ndata['mask_emb'] = ent_mask
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            kg.edata['mask_emb'] = rel_mask[rel_id]
            kg.edata['emb'] = kg.edata['emb'].type(torch.float32)
            kg.edata['mask_emb'] = kg.edata['mask_emb'].type(torch.float32)
            kg.ndata['emb'] = kg.ndata['emb'].type(torch.float32)
            kg.ndata['mask_emb'] = kg.ndata['mask_emb'].type(torch.float32)
            if self.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'emb'))
                kg.apply_edges(fn.u_add_e('mask_emb', 'mask_emb', 'm_mask'))
                kg.apply_edges(fn.e_sub_v('emb', 'emb', 'comp_emb'))
                kg.apply_edges(fn.e_add_v('m_mask', 'mask_emb', 'comp_mask'))
            elif self.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'emb'))
                kg.apply_edges(fn.u_add_e('mask_emb', 'mask_emb', 'mask_emb'))
                kg.apply_edges(fn.e_mul_v('emb', 'emb', 'comp_emb'))
                kg.apply_edges(fn.e_add_v('mask_emb', 'mask_emb', 'comp_mask'))
            else:
                raise NotImplementedError
            comp_mask = kg.edata['comp_mask']
            comp_mask[comp_mask > 1] = 1
            kg.edata['comp_mask'] = comp_mask
            query = question_emb.unsqueeze(1).to(device)
            comp_emb = kg.edata['comp_emb']
            query_emb = self.tok_linear(query)
            key_emb = self.key_linear(comp_emb).permute(0, 2, 1)
            weight = self.scale * (torch.matmul(query_emb, key_emb))
            kg_mask = kg.edata['mask_emb'].to(torch.float32)
            kg_mask = kg_mask[None, :, None, :]
            mask = (kg_mask != 0).float()
            epsilon = torch.sum(mask, dim=3)
            epsilon = torch.where(epsilon == 0, torch.ones_like(epsilon), epsilon)
            weight = torch.sum(weight * kg_mask, dim=3) / epsilon
            weight = torch.mean(weight, dim=-1)
            atts = stable_softmax(weight)
            kg.edata['comp_emb'] = self.comp_linear(key_emb).squeeze(2)
            neigh_ent_emb =[]
            for att in atts:
                kg = kg.to(device)
                kg.edata['weight'] = att.unsqueeze(1)
                kg.edata['weight'] = kg.edata['weight'].to(torch.float32)
                kg = kg.to('cpu')
                sample_kg = dgl.sampling.select_topk(kg, 2, 'weight', edge_dir='out')
                sample_kg = sample_kg.to(device)
                sample_kg.edata['comp_emb_att'] = sample_kg.edata['comp_emb'] * sample_kg.edata['weight']
                for i in range(self.skip):
                    sample_kg.update_all(fn.copy_e('comp_emb_att', 'm'), fn.sum('m', 'neigh'))
                    if self.skip > 1:
                        sample_kg.apply_edges(fn.u_add_v('neigh', 'neigh', 'comp_emb_att'))
                neight = sample_kg.ndata['neigh']
                neigh_ent_emb.append(neight)
            neigh_ent_embs = torch.stack(neigh_ent_emb)
            neigh_embs_k = self.attentionKGselector(neigh_ent_embs)
            kg_emb = self.act(self.kg_linear(neigh_embs_k))
        return kg_emb

def stable_softmax(x, dim=-1):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_val)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class AttentionBasedKnowledgeSelector(nn.Module):
    def __init__(self, hidden_size, target_dim):
        super(AttentionBasedKnowledgeSelector, self).__init__()
        self.hidden_size = hidden_size
        self.target_dim = target_dim
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = (hidden_size ** -0.5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        self.query_proj.bias.data.fill_(0)
        self.key_proj.bias.data.fill_(0)
        self.value_proj.bias.data.fill_(0)

    def forward(self, knowledge_feat):
        knowledge_feat = F.layer_norm(knowledge_feat, normalized_shape=[self.hidden_size])
        Q = self.query_proj(knowledge_feat)
        K = self.key_proj(knowledge_feat)
        V = self.value_proj(knowledge_feat)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_knowledge = torch.matmul(attention_weights, V)
        importance_scores = attention_weights.mean(dim=1)
        topk_scores, topk_indices = torch.topk(importance_scores, self.target_dim, dim=-1)
        assert topk_indices.max() < attended_knowledge.size(1)
        selected_knowledge = torch.gather(
            attended_knowledge, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        return selected_knowledge

def compute_corr(x1, x2):
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)
    return corr

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        y = self.net[-1](x)
        return y

class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)

    def forward(self, x1, x2):
        y1 = self.phi(x1)
        y2 = self.psi(x2)
        return y1, y2

def construct_dict(dir_path, set_flag):
    ent2id, rel2id = dict(), dict()
    ents, rels = [], []
    path = join(dir_path, '{}.txt'.format(set_flag))
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if '#' in line:
                parts = line.strip().split('#')
                h, r, t = '', '', ''
                if len(parts) == 3:
                    h, r, t = parts
                t = t[:-1]
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                    ents.append(h)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                    ents.append(t)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
                    rels.append(r)
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    return ent2id, rel2id, ents, rels

def read_data(set_flag):
    assert set_flag in ['slake_kg', 'RadLex']
    dir_p = r"data/"
    ent2id, rel2id, ents, rels = construct_dict(dir_p, set_flag)
    if set_flag in ['slake_kg', 'RadLex']:
        path = join(dir_p, '{}.txt'.format(set_flag))
        file = open(path, 'r', encoding='utf-8')
    else:
        raise NotImplementedError
    src_list = []
    dst_list = []
    rel_list = []
    for i, line in enumerate(file):
        if '#' in line:
            parts = line.strip().split('#')
            h, r, t = '', '' ,''
            if len(parts) == 3:
                h, r, t = parts
            t = t[:-1]
            h, r, t = ent2id[h], rel2id[r], ent2id[t]
            src_list.append(h)
            dst_list.append(t)
            rel_list.append(r)
    file.close()
    output_dict = {
        'ent2id': ent2id,
        'rel2id': rel2id,
        'ents': ents,
        'rels': rels,
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
    }
    return output_dict

def construct_kg(set_flag, n_rel, directed=False):
    assert directed in [True, False]
    d = read_data(set_flag)
    src_list, dst_list, rel_list = [], [], []
    eid = 0
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        if directed:
            src_list.extend([h])
            dst_list.extend([t])
            rel_list.extend([r])
            eid += 1
        else:
            src_list.extend([h, t])
            dst_list.extend([t, h])
            rel_list.extend([r, r + n_rel])
            eid += 2
    src, dst, rel = torch.tensor(src_list), torch.tensor(dst_list), torch.tensor(rel_list)
    return src, dst, rel

def get_kg(src, dst, rel, n_ent, device):
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    kg.edata['rel_id'] = rel
    kg = kg.to(device)
    return kg

def encode_kg(ents, rels):
    tokenizer = AutoTokenizer.from_pretrained("pre-model/biobert_v1.1")
    ents_emb, rels_emb, ents_mask, rels_mask = [], [], [], []
    for ent in ents:
        ent_word = tokenizer.tokenize(ent)
        ent_emb = tokenizer.convert_tokens_to_ids(ent_word)
        ent_emb = ent_emb[:16]
        ent_mask = [1] * len(ent_emb)
        n_pad = 16 - len(ent_emb)
        ent_emb.extend([0] * n_pad)
        ent_mask.extend([0] * n_pad)
        ents_emb.append(ent_emb)
        ents_mask.append(ent_mask)
    ents_emb = torch.tensor(ents_emb, dtype=torch.float, requires_grad=True).to(device)
    ents_mask = torch.tensor(ents_mask, dtype=torch.float, requires_grad=True).to(device)
    for rel in rels:
        rel_word = tokenizer.tokenize(rel)
        rel_emb = tokenizer.convert_tokens_to_ids(rel_word)
        rel_emb = rel_emb[:16]
        rel_mask = [1] * len(rel_emb)
        n_pad = 16 - len(rel_emb)
        rel_emb.extend([0] * n_pad)
        rel_mask.extend([0] * n_pad)
        rels_emb.append(rel_emb)
        rels_mask.append(rel_mask)
    rels_emb = torch.tensor(rels_emb, dtype=torch.float, requires_grad=True).to(device)
    rels_mask = torch.tensor(rels_mask, dtype=torch.float, requires_grad=True).to(device)
    return ents_emb, rels_emb, ents_mask, rels_mask

def get_kg_emb(set_flag):
    d = read_data(set_flag)
    ent_emb, rel_emb, ent_mask, rel_mask = encode_kg(d['ents'], d['rels'])
    return ent_emb, rel_emb, ent_mask, rel_mask
