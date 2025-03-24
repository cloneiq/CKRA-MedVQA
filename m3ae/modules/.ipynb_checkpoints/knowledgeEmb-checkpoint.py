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
        # self.ent_linear = nn.Linear(768, args.h_dim, bias=False)

        # self.rel_linear = nn.Linear(768, args.h_dim, bias=False)

        # KG
        self.n_ent = 1968
        self.n_rel = 58

        self.ent_emb, self.b_rel_emb, self.ent_mask, self.b_rel_mask = get_kg_emb("slake_kg")
        self.kg_n_layer = 2
        self.comp_layers = nn.ModuleList([CompLayer('add', hidden_size) for _ in range(self.kg_n_layer)])

        # 乘2因为无向图边是双向的
        self.rel_embs = nn.ParameterList([torch.cat((self.b_rel_emb, self.b_rel_emb), dim=0) for _ in range(self.kg_n_layer)])
        self.rel_mask = torch.cat((self.b_rel_mask, self.b_rel_mask), dim=0)
        self.rel_w = get_param(hidden_size, hidden_size)
        self.ent_drop = nn.Dropout(0.2)
        self.act = nn.Tanh()       # nn.LeakyReLU()
        self.L = nn.Linear(hidden_size, hidden_size)
        self.S = nn.Linear(hidden_size, hidden_size)
        self.mea_func = Measure_F(hidden_size, hidden_size, [200] * 2, [200] * 2)
        # self.kg_linear = nn.Linear(int(self.n_ent / args.top_k), 20, bias=False)
        # self.kg_w = get_param(args.top_k, 20)
        self.ent_emb_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.top_k = int(self.n_ent * 0.1)
        self.attentionKGselector = AttentionBasedKnowledgeSelector(hidden_size=hidden_size, target_dim=self.top_k)
#         self.PCA_Layer = PCALayer(self.top_k, 32)


    def forward(self, kg, question_embedding):
        ent_emb = self.ent_emb.long()
        ent_emb = self.ent_embedding(ent_emb)
        # ent_emb = self.ent_linear(ent_emb)
        common = self.act(self.S(ent_emb))
        private = self.act(self.L(ent_emb))
        rel_emb_list = []

        ent_emb = torch.mean(ent_emb, dim=1)
        ent_emb = self.ent_emb_linear(ent_emb)
        ent_emb = torch.stack([ent_emb] * question_embedding.shape[0], dim=0)
        ent_emb = self.attentionKGselector(ent_emb)

        for comp_layer, rel_emb in zip(self.comp_layers, self.rel_embs):
            rel_emb = rel_emb.long()
            rel_emb = self.ent_embedding(rel_emb)
            # rel_emb = self.rel_linear(bert_rel_emb)

            ent_emb = self.ent_drop(ent_emb)

            # comp_layer就是一个CompLayer类的实例化
            comp_ent_emb1 = comp_layer(kg, common, rel_emb, self.ent_mask, self.rel_mask, question_embedding)
            comp_ent_emb2 = comp_layer(kg, private, rel_emb, self.ent_mask, self.rel_mask, question_embedding)

            ent_emb = ent_emb + comp_ent_emb1 + comp_ent_emb2

            rel_emb_list.append(rel_emb)
            phi_c, phi_p = self.mea_func(comp_ent_emb1, comp_ent_emb2)
            corr = compute_corr(phi_c, phi_p)


        # kg_emb = self.act(self.kg_linear(ent_emb.permute(0, 2, 1)).permute(0, 2, 1))
        # print("ent_emb:",ent_emb.shape)
        # kg_emb = self.act(torch.matmul(ent_emb.permute(0, 2, 1),self.kg_w).permute(0, 2, 1))
        kg_emb = self.act(ent_emb)
        # print("kg_emb:",kg_emb.shape)

        return kg_emb, corr

class PCALayer(nn.Module):
    def __init__(self, num_ent, reduced_dim):
        """
        :param num_ent: 输入特征的第二维大小
        :param reduced_dim: 降维后的第二维大小
        """
        super(PCALayer, self).__init__()
        # 定义一个投影矩阵，形状为 [num_ent, reduced_dim]
        self.proj_matrix = nn.Parameter(torch.randn(num_ent, reduced_dim))
        self._orthogonalize()

    def _orthogonalize(self):
        """
        确保投影矩阵保持正交性
        """
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.proj_matrix)
            self.proj_matrix.copy_(Q)

    def forward(self, x):
        """
        :param x: 输入特征 [batch_size, num_ent, hidden_size]
        :return: 降维后的特征 [batch_size, reduced_dim, hidden_size]
        """
        # 数据中心化：对输入数据进行均值化
        # 对 batch 维度进行均值化
        x_centered = x - x.mean(dim=1, keepdim=True)  # [batch_size, num_ent, hidden_size]

        # 对 num_ent 维度进行线性投影
        # x: [batch_size, num_ent, hidden_size]
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
        self.hidden_size = hidden_size
        self.neigh_w = get_param(self.hidden_size, self.hidden_size)
        self.act = nn.Tanh()  # nn.ReLU()
        self.hidden_size = self.hidden_size
        self.tok_linear = nn.Linear(self.hidden_size, self.h_dim)
        self.key_linear = nn.Linear(self.h_dim, self.h_dim)
        self.kg_linear = nn.Linear(self.h_dim, self.hidden_size)
        # self.norm_layer = nn.LayerNorm(self.h_dim, eps=1e-12)
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
            
            # 转换边特征的数据类型
            kg.edata['emb'] = kg.edata['emb'].type(torch.float32)
            kg.edata['mask_emb'] = kg.edata['mask_emb'].type(torch.float32)
    
            # 转换节点特征的数据类型
            kg.ndata['emb'] = kg.ndata['emb'].type(torch.float32)
            kg.ndata['mask_emb'] = kg.ndata['mask_emb'].type(torch.float32)

            # # u头，v尾，e边

            # 第一次聚合：邻居信息的聚合
            # 第二次聚合：邻居的邻居信息的聚合
            if self.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'emb'))
                kg.apply_edges(fn.u_add_e('mask_emb', 'mask_emb', 'm_mask'))    # 保留掩码
                kg.apply_edges(fn.e_add_v('emb', 'emb', 'comp_emb'))
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
            # attention
            # 计算点积
            query = question_emb.unsqueeze(1).to(device)  # torch.Size([batch_size, 1, 32, 768])
            comp_emb = kg.edata['comp_emb']

            query_emb = self.tok_linear(query)
            key_emb = self.key_linear(comp_emb).permute(0, 2, 1)
            weight = self.scale * (torch.matmul(query_emb, key_emb))
#             print(weight.shape)  torch.Size([2, 5200, 32, 16])

            kg_mask = kg.edata['mask_emb'].to(torch.float32)
            kg_mask = kg_mask[None, :, None, :]
            mask = (kg_mask != 0).float()
            epsilon = torch.sum(mask, dim=3)
            epsilon = torch.where(epsilon == 0, torch.ones_like(epsilon), epsilon)  # 避免分母为零
            weight = torch.sum(weight * kg_mask, dim=3) / epsilon
            weight = torch.mean(weight, dim=-1)
#             print(weight.shape)  [2, 5200]
            # print(weight.shape)
            # question_mask = question_mask[:, None, :]
            # mask = (question_mask != 0).float()
            # epsilon = torch.sum(mask, dim=2)
            # epsilon = torch.where(epsilon == 0, torch.ones_like(epsilon), epsilon)  # 避免分母为零
            # weight = torch.sum(weight * question_mask, dim=2) / epsilon

            atts = stable_softmax(weight)

            kg.edata['comp_emb'] = self.comp_linear(key_emb).squeeze(2)
            neigh_ent_emb =[]
            for att in atts:
                kg = kg.to(device)
                kg.edata['weight'] = att.unsqueeze(1)

                kg.edata['weight'] = kg.edata['weight'].to(torch.float32)
                # kg.edata['weight'] = kg.edata['norm']
                kg = kg.to('cpu')
                # top-k采样    采样每个节点为尾节点的前k个重要的三元组，in是度，out是出度
                #     sample_kg = dgl.sampling.select_topk(kg, self.k, 'weight', edge_dir='in')
                sample_kg = dgl.sampling.select_topk(kg, 4, 'weight', edge_dir='out')    # 一样的,因为是无向图
                #     sample_kg = kg
                sample_kg = sample_kg.to(device)

                # agg
#                 print(sample_kg.edata['comp_emb'].shape)
#                 print(sample_kg.edata['weight'].shape)
#                 torch.Size([3433, 768])
#                 torch.Size([3433, 1, 32])
#                 sys.exit()
                sample_kg.edata['comp_emb_att'] = sample_kg.edata['comp_emb'] * sample_kg.edata['weight']
                for i in range(self.skip):
                    sample_kg.update_all(fn.copy_e('comp_emb_att', 'm'), fn.sum('m', 'neigh'))
                    if self.skip > 1:
                        sample_kg.apply_edges(fn.u_add_v('neigh', 'neigh', 'comp_emb_att'))
                neight = sample_kg.ndata['neigh']    # .mm(self.neigh_w)
                neigh_ent_emb.append(neight)
            neigh_ent_embs = torch.stack(neigh_ent_emb)
            neigh_embs_k = self.attentionKGselector(neigh_ent_embs)

            # 激活函数激活
            kg_emb = self.act(self.kg_linear(neigh_embs_k))

        return kg_emb

# 稳定归一化
def stable_softmax(x, dim=-1):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)  # 计算每个dim最大值
    x_exp = torch.exp(x - max_val)  # 减去最大值避免梯度爆炸
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

# 自注意力计算权重，按权重选择最相关的 1/k 个知识
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

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # [2, 1968, 1968]
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_knowledge = torch.matmul(attention_weights, V)
        importance_scores = attention_weights.mean(dim=1)  # [2, 1968]
        topk_scores, topk_indices = torch.topk(importance_scores, self.target_dim, dim=-1)

        assert topk_indices.max() < attended_knowledge.size(1)
        selected_knowledge = torch.gather(
            attended_knowledge, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )

        return selected_knowledge

def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

    return corr


class MLP(nn.Module):
    #  h_dim, [200]*2, 1
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        # 初始化父类
        super(MLP, self).__init__()
        # 创建一个ModuleList对象，用于存放多个层
        self.net = nn.ModuleList()
        # 实例化Dropout层，dropprob为丢弃概率
        self.dropout = torch.nn.Dropout(dropprob)
        # 定义网络的结构，input_d为输入维度，structure为隐藏层结构，output_d为输出维度
        struc = [input_d] + structure + [output_d]
        # print(struc)   # [512, 200, 200, 1]
        # sys.exit()

        # 循环遍历结构列表，创建并添加线性层到self.net
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        # 循环遍历self.net，除了最后一个层
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            # 应用Dropout层
            x = self.dropout(x)

        # 对于最后一个层
        y = self.net[-1](x)

        return y


class Measure_F(nn.Module):
    #        (h_dim, h_dim, [200]*2, [200]*2)
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)

    def forward(self, x1, x2):
        y1 = self.phi(x1)
        y2 = self.psi(x2)
        return y1, y2

# class MLP1(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP1, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y
#
#
# class MLP2(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP2, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y


# measurable functions \phi and \psi


# 从文件中读取数据以构建实体和关系的字典，为每个实体和关系分配一个唯一的ID。
def construct_dict(dir_path, set_flag):
    """
    construct the entity, relation dict
    :param dir_path: data directory path
    :return:
    """
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
                t = t[:-1]  # remove \n
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                    ents.append(h)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                    ents.append(t)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
                    rels.append(r)

    # print(len(ents), len(rels))  # 1968, 58  # RadLex：163787 26
    # sys.exit("结束")

    # 根据值对字典排序，排序后重新转换成字典
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    return ent2id, rel2id, ents, rels


def read_data(set_flag):

    assert set_flag in ['slake_kg', 'RadLex']
    dir_p = r"data/"
    ent2id, rel2id, ents, rels = construct_dict(dir_p, set_flag)

    if set_flag in ['slake_kg', 'RadLex']:
        path = join(dir_p, '{}.txt'.format(set_flag))
        file = open(path, 'r', encoding='utf-8')
    # 如果打开多个文件，将其串联起来
    # if set_flag == ['train', 'valid', 'test']:
    #     path1 = join(dir_p, 'train.txt')
    #     path2 = join(dir_p, 'valid.txt')
    #     path3 = join(dir_p, 'test.txt')
    #     file1 = open(path1, 'r', encoding='utf-8')
    #     file2 = open(path2, 'r', encoding='utf-8')
    #     file3 = open(path3, 'r', encoding='utf-8')
    #     file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    src_list = []
    dst_list = []
    rel_list = []

    # 默认值为set（集合）的字典,集合是一种无序且不包含重复元素的数据结构，因此可以用来存储多个尾部实体，而不会重复
    # 三个字典的键代表ID或名称，而值则是集合，集合中包含关系的具体信息（如头部实体、尾部实体、关系类型等）
    # pos_tails = defaultdict(set)   # 存储"每个关系"的尾部实体
    # pos_heads = defaultdict(set)   # 存储"每个关系"的头部实体
    # pos_rels = defaultdict(set)    # 存储"每个关系"的类型

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

            # 保存连接信息（图结构）
            # pos_tails[(h, r)].add(t)     # {(h, r): {t}}
            # pos_heads[(r, t)].add(h)
            # pos_rels[(h, t)].add(r)  # edge relation
            # pos_rels[(t, h)].add(r+len(rel2id))  # reverse relations  反向关系

    file.close()

    output_dict = {
        'ent2id': ent2id,
        'rel2id': rel2id,
        'ents': ents,
        'rels': rels,
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        # 'pos_tails': pos_tails,
        # 'pos_heads': pos_heads,
        # 'pos_rels': pos_rels
    }

    # 返回实体关系的ID列表, 实体关系集，三元组ID数组（三个数组，相同位置对应一个三元组）
    return output_dict


# 将图数据转换为适合图神经网络（Graph Neural Networks, GNN）或其他图算法处理的形式。
# 通过将节点和关系映射到唯一的边ID，并存储在hr2eid和rt2eid字典中，可以方便地查询和操作图中的边
def construct_kg(set_flag, n_rel, directed=False):
    """
    construct kg.
    :param set_flag: train / valid / test set flag, use which set data to construct kg.
    :param directed: whether add inverse version for each edge, to make a undirected graph.
    :return:
    """
    assert directed in [True, False]

    # 实体关系的ID列表, 实体关系集，三元组ID数组（三个数组，相同位置对应一个三元组）
    d = read_data(set_flag)
    src_list, dst_list, rel_list = [], [], []

    eid = 0   # 边计数器
    # 定义两个默认为"list"的字典，会自动为不存在的键创建一个默认值（这里是列表）
    # hr2eid, rt2eid = defaultdict(list), defaultdict(list)
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        # 有向图
        if directed:
            # list 的 extend 方法用于将一个可迭代对象（如列表、元组、集合等）中的所有元素添加到当前列表的末尾
            src_list.extend([h])
            dst_list.extend([t])
            rel_list.extend([r])
            # hr2eid[(h, r)].extend([eid])
            # rt2eid[(r, t)].extend([eid])
            eid += 1
        # 无向图，要添加反向的边
        else:
            # include the inverse edges
            # inverse rel id: original id + rel num
            src_list.extend([h, t])  # [h, t, h, t,...]
            dst_list.extend([t, h])  # [t, h, t, h,...]
            rel_list.extend([r, r + n_rel])
            # hr2eid[(h, r)].extend([eid, eid + 1])
            # rt2eid[(r, t)].extend([eid, eid + 1])
            eid += 2

    src, dst, rel = torch.tensor(src_list), torch.tensor(dst_list), torch.tensor(rel_list)

    # print("rel: ", rel.shape)    # torch.Size([4434])   2217个三元组，2217*2 = 4434
    # print(src.shape, dst.shape)    # torch.Size([5200]) torch.Size([5200])
    return src, dst, rel


def get_kg(src, dst, rel, n_ent, device):
    # 创建一个知识图谱
    # 例如，如果src = [0, 1, 2]，dst = [1, 2, 3]，num_nodes = 4，那么创建的图将包含3条边：0->1，1->2，2->3，并且图中总共有4个节点
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    # 添加关系数据，
    kg.edata['rel_id'] = rel
    # 将图移动到指定设备
    kg = kg.to(device)
    # 返回图
    return kg


def encode_kg(ents, rels):
    # print(ents[0])    # disease
    # tokenizer = BertTokenizer.from_pretrained(bert_base)
    ents_emb, rels_emb, ents_mask, rels_mask = [], [], [], []
    tokenizer = AutoTokenizer.from_pretrained("pre-model/biobert_v1.1")
    for ent in ents:  # 1327次
        ent_word = tokenizer.tokenize(ent)    # ['disease']
        ent_emb = tokenizer.convert_tokens_to_ids(ent_word)       # [3653]
        ent_emb = ent_emb[:16]        # [3653]
        ent_mask = [1] * len(ent_emb)
        n_pad = 16 - len(ent_emb)
        ent_emb.extend([0] * n_pad)  # [3653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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


# 替换掉ent, rel的初始化矩阵，用预训练的词向量
def get_kg_emb(set_flag):
    d = read_data(set_flag)
    ent_emb, rel_emb, ent_mask, rel_mask = encode_kg(d['ents'], d['rels'])
    # print(ent_emb.shape, rel_emb.shape, ent_mask.shape, rel_mask.shape)
    # torch.Size([1327, 16]) torch.Size([13, 16]) torch.Size([1327, 16]) torch.Size([13, 16])

    return ent_emb, rel_emb, ent_mask, rel_mask