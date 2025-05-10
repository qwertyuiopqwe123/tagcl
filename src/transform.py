import copy
import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose
from src.utils import  feature_drop_weights, drop_feature_weighted_2,  pseudo_drop_weights,cal_Weights
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def other_transform(args,data,device, predict_lbl_pro, degree_sim,  weights, Z=None):
        drop_feature_rate_1=args.drop_feature_rate_1
        drop_edge_rate_1=args.drop_edge_rate_1
        node_w_ps,new_index, new_prediction=cal_Weights(Z, predict_lbl_pro, degree_sim, device, weights)
        node_w_ps=node_w_ps.to(device)
        drop_weights = pseudo_drop_weights(data, node_w_ps).to(device)
        node_pseduo = degree_sim.float().to(device)
        feature_weights = feature_drop_weights(data.x, node_c=node_pseduo).to(device)
        edge_index1 = drop_edge_weighted(data.edge_index, drop_weights, drop_edge_rate_1, threshold=0.7)

        x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
        return edge_index1, x_1,new_index, new_prediction

def gdc(A: sp.csr_matrix, alpha=0.5, eps=0.0001):
    N = A.shape[0]
    A_loop = sp.eye(N, format='csr') + A  # 确保是 CSR 格式
    D_loop_vec = A_loop.sum(0).A1  # 获取度数向量
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)  # 创建对角矩阵
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt  # 计算对称传播矩阵
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)  # 计算传播矩阵
    S_tilde = S.multiply(S >= eps)  # 去除小值
    D_tilde_vec = S_tilde.sum(0).A1  # 获取新的度数向量
    T_S = S_tilde / D_tilde_vec  # 归一化
    return sp.csr_matrix(T_S)

def create_diffusion_graph(A: sp.csr_matrix, x: torch.Tensor, device,alpha=0.5, eps=0.0001):
    # 保持 A 在 GPU 上进行稀疏运算，避免转换为 NumPy 数组
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()  # 在将 A 转换为稀疏矩阵时，将其转移到 CPU
    A_csr = sp.csr_matrix(A)  # 转换为 CSR 格式稀疏矩阵
    # 使用 gdc 计算扩散矩阵（假设 gdc 返回稀疏矩阵）
    T_S = gdc(A_csr, alpha=alpha, eps=eps)
    # 保持 T_S 为稀疏矩阵，避免转为稠密矩阵
    coo = T_S.tocoo()  # 保持稀疏表示
    # 转换为 PyTorch 的稀疏张量
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    T_S_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))
    # 计算更新后的特征，使用稀疏矩阵乘法
    updated_features = torch.sparse.mm(T_S_tensor, x)
    # 创建边索引
    edge_index = torch.tensor(np.vstack((T_S.nonzero())), dtype=torch.long, device=device)
    return edge_index, updated_features


def create_pseudo_label_graph(x: torch.Tensor, device, similarity_threshold=0.5):
    """
    创建伪标签图，并基于伪标签图更新特征
    :param x: 节点特征 (n x d)，通常为 PyTorch 张量
    :param device: 计算设备（如 'cuda' 或 'cpu'）
    :param similarity_threshold: 节点之间的相似性阈值，用于构建边
    :return: edge_index（边索引）, updated_features（更新后的特征）
    """
    # 对密集张量进行 L2 归一化
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)  # 对每行进行归一化
    # 计算特征相似性矩阵（余弦相似度）
    similarity_matrix = torch.mm(x_norm, x_norm.T)  # (n x n) 相似性矩阵
    # 根据相似性阈值创建邻接矩阵
    if isinstance(similarity_threshold, str):
        similarity_threshold = float(similarity_threshold)
        # 只保留相似性大于阈值的边
    adj_matrix = (similarity_matrix > similarity_threshold).float()  # 只保留相似性大于阈值的边
    # 转换为稀疏表示（COO 格式）
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).T  # 边的索引 (2 x num_edges)
    values = similarity_matrix[edge_index[0], edge_index[1]]  # 边的权重
    sparse_adj = torch.sparse.FloatTensor(edge_index, values, adj_matrix.size()).to(device)
    # 使用稀疏矩阵乘法更新特征
    updated_features = torch.sparse.mm(sparse_adj, x.to(device))

    # 返回边索引和更新特征
    return edge_index, updated_features



def label_smoothing_graph(A: sp.csr_matrix, x: torch.Tensor, device, smoothing=0.1):
    """
    基于标签平滑思想构造新图（修复稀疏矩阵加标量的问题）。

    :param A: 原始邻接矩阵（scipy 稀疏矩阵格式）。
    :param x: 节点特征张量，形状为 (num_nodes, num_features)。
    :param device: 设备信息（'cuda' 或 'cpu'）。
    :param smoothing: 平滑因子 ε，用于平滑邻接矩阵。
    :return: 平滑后的边索引 (edge_index) 和更新后的节点特征 (updated_features)。
    """
    # 将邻接矩阵转为概率转移矩阵（行归一化）
    row_sum = np.array(A.sum(1)).flatten()  # 每行的度数
    row_inv = np.power(row_sum, -1, where=row_sum > 0)  # 度的逆
    row_inv[np.isinf(row_inv)] = 0  # 处理无连接节点
    D_inv = sp.diags(row_inv)  # 构造对角矩阵
    P = D_inv @ A  # 归一化后的概率转移矩阵

    # 构造稀疏均匀分布矩阵
    num_nodes = A.shape[0]
    uniform_matrix = sp.csr_matrix((np.ones(num_nodes) / num_nodes, (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes))

    # 平滑矩阵
    smoothed_A = P.multiply(1 - smoothing) + uniform_matrix.multiply(smoothing)

    # 保持稀疏表示
    coo = smoothed_A.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    T_S_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))

    updated_features = []
    batch_size = 1000  # 每次处理 1000 个节点
    num_nodes = x.shape[0]

    for i in range(0, num_nodes, batch_size):
        # 当前批次的行索引范围
        batch_rows = range(i, min(i + batch_size, num_nodes))

        # 提取子稀疏矩阵对应的行
        row_mask = torch.tensor(batch_rows, dtype=torch.long, device=device)
        sub_indices = T_S_tensor._indices()
        sub_values = T_S_tensor._values()

        # 筛选属于当前批次的行
        mask = (sub_indices[0, :] >= i) & (sub_indices[0, :] < i + batch_size)
        filtered_indices = sub_indices[:, mask]
        filtered_values = sub_values[mask]

        # 调整行索引，使其从 0 开始
        filtered_indices[0, :] -= i

        # 构造子稀疏矩阵
        sub_T_S_tensor = torch.sparse.FloatTensor(
            filtered_indices, filtered_values, torch.Size((len(batch_rows), num_nodes))
        )

        # 批次特征计算
        batch_x = x
        updated_batch = torch.sparse.mm(sub_T_S_tensor, batch_x)
        updated_features.append(updated_batch)

    # 合并结果
    updated_features = torch.cat(updated_features, dim=0)

    # 创建边索引
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)

    return edge_index, updated_features


def create_attention_graph(A: sp.csr_matrix, x: torch.Tensor, device, alpha=0.2, dropout=0.0):
    """
    使用自注意力机制生成扩散图，并更新节点特征。

    :param A: 原始邻接矩阵（scipy csr_matrix 格式）
    :param x: 节点特征张量，形状为 (num_nodes, num_features)
    :param device: 设备信息（'cuda' 或 'cpu'）
    :param alpha: LeakyReLU 的负斜率
    :param dropout: Dropout 的概率
    :return: 平滑后的边索引 (edge_index) 和更新后的节点特征 (updated_features)
    """
    # 1. 转换邻接矩阵为 COO 格式
    coo = A.tocoo()
    row, col = coo.row, coo.col

    # 2. 计算注意力权重
    num_nodes, num_features = x.size()
    x_row = x[row]  # 邻接矩阵中每条边起始节点的特征
    x_col = x[col]  # 邻接矩阵中每条边目标节点的特征

    # 特征拼接并计算注意力
    edge_features = torch.cat([x_row, x_col], dim=1)  # 每条边的特征 (num_edges, 2 * num_features)
    attention_mlp = nn.Linear(2 * num_features, 1, bias=False).to(device)
    nn.init.xavier_uniform_(attention_mlp.weight, gain=1.414)

    e = F.leaky_relu(attention_mlp(edge_features), negative_slope=alpha)  # (num_edges, 1)
    e = e.squeeze(1)  # (num_edges,)

    # 3. 应用 softmax 归一化
    edge_scores = torch.zeros_like(e).to(device)  # 初始化归一化权重
    edge_scores[row] = torch.exp(e)
    row_sum = torch.zeros(num_nodes, device=device).index_add_(0, torch.tensor(row, device=device), edge_scores)  # 行归一化
    attention_weights = edge_scores / row_sum[row]  # softmax

    # 4. 构造稀疏矩阵
    indices = torch.tensor([row, col], dtype=torch.long, device=device)
    attention_matrix = torch.sparse.FloatTensor(indices, attention_weights, torch.Size([num_nodes, num_nodes]))

    # 5. 稀疏矩阵乘法更新节点特征
    updated_features = torch.sparse.mm(attention_matrix, x)

    # 6. 构造边索引
    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long, device=device)

    return edge_index, updated_features


def label_propagation(A: sp.csr_matrix, max_iter=100):
    """
    使用标签传播算法生成扩散图邻接矩阵。

    :param A: 输入的稀疏邻接矩阵 (scipy csr_matrix 格式)。
    :param max_iter: 最大迭代次数。
    :return: 稀疏的传播矩阵 (scipy csr_matrix 格式)。
    """
    num_nodes = A.shape[0]

    # 初始化标签：每个节点的标签为其索引
    labels = np.arange(num_nodes)

    # 获取邻接矩阵的行索引和列索引
    row, col = A.nonzero()

    # 标签传播迭代
    for _ in range(max_iter):
        updated = False
        for i in range(num_nodes):
            # 找到节点 i 的邻居节点
            neighbors = col[row == i]
            if len(neighbors) == 0:
                continue

            # 统计邻居标签频率
            neighbor_labels = labels[neighbors]
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]

            # 更新节点 i 的标签
            if labels[i] != most_common_label:
                labels[i] = most_common_label
                updated = True

        # 如果标签不再变化，则停止迭代
        if not updated:
            break

    # 构造新的邻接矩阵
    new_rows, new_cols, new_data = [], [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if labels[i] == labels[j]:
                new_rows.append(i)
                new_cols.append(j)
                new_data.append(1.0)

    return sp.csr_matrix((new_data, (new_rows, new_cols)), shape=(num_nodes, num_nodes))


def create_diffusion_graph_with_lpa(A: sp.csr_matrix, x: torch.Tensor, device, max_iter=100):
    """
    使用 LPA 和扩散思想生成新图。

    :param A: 原始邻接矩阵 (scipy csr_matrix)。
    :param x: 节点特征张量，形状为 (num_nodes, num_features)。
    :param device: 设备信息（'cuda' 或 'cpu'）。
    :param max_iter: LPA 的最大迭代次数。
    :return: 平滑后的边索引 (edge_index) 和更新后的节点特征 (updated_features)。
    """
    # 1. 使用 LPA 生成新邻接矩阵
    T_S = label_propagation(A, max_iter=max_iter)

    # 2. 保持 T_S 为稀疏矩阵，转换为 PyTorch 稀疏张量
    coo = T_S.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    T_S_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))

    # 3. 使用稀疏矩阵乘法更新节点特征
    updated_features = torch.sparse.mm(T_S_tensor, x)

    # 4. 创建边索引
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)

    return edge_index, updated_features