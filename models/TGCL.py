import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from embedder import embedder
from src.sampling import Sampler
from src.utils import reset, set_random_seeds, masking,similarity
from copy import deepcopy
from src.transform import other_transform
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy
from src.transform import create_diffusion_graph , create_pseudo_label_graph,label_smoothing_graph,create_attention_graph,create_diffusion_graph_with_lpa
import scipy.sparse as sp

import numpy as np
class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def similar_loss(h1, h2):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, h2))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

class TGCL_Trainer(embedder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def getadj(self, data, device):

        edge_index = data.edge_index.to(device)  # 确保 edge_index 在正确的设备上
        num_nodes = data.num_nodes  # 节点数量

        # 直接使用 PyTorch 创建 COO 稀疏矩阵
        row = edge_index[0].cpu().numpy()  # 从 GPU 转回 CPU 以进行 NumPy 操作
        col = edge_index[1].cpu().numpy()  # 从 GPU 转回 CPU 以进行 NumPy 操作
        adj = sp.coo_matrix((np.ones(edge_index.size(1)), (row, col)), shape=(num_nodes, num_nodes))

        # 转换为 CSR 格式
        adj_csr = adj.tocsr()
        return adj_csr
    def teacher_update_ma(self):
        assert self.model_gnnt is not None, 'teacherE encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.model_gnnt, self.model_gnn)
    def sim(self,h1, h2):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        return torch.mm(z1, z2.t())

    def similar_loss(self,h1, h2):
        f = lambda x: torch.exp(x)
        cross_sim = f(self.sim(h1, h2))
        return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))
    def _init_model(self):
        self.model_gnn = TGCL(self.encoder, self.classifier, self.unique_labels, self.args.tau, self.args.thres, self.args.device,flag=1).to(self.device)
        self.model_gnnt = copy.deepcopy(self.model_gnn)
        self.moving_average_decay = 0.999
        self.model_gnnt.flag = 0  # 更改标志，表示是 gnn 教师模型
        # 禁用教师模型的参数梯度更新
        for param in self.model_gnnt.parameters():
            param.requires_grad = False
        self.optimizer = Adam(self.model_gnn.parameters(), lr=self.args.lr, weight_decay=self.args.decay)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1])
        self.weights = node_deg / max(node_deg)
        self.nclass = self.data.y.max().item() + 1
        self.teacher_ema_updater = EMA(self.moving_average_decay)  # 假设有一个 EMA 更新器


    def _init_dataset(self):
        self.labels = deepcopy(self.data.y)
        self.running_train_mask = deepcopy(self.train_mask)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1])
        self.degree_sim, self.predict_lbl_pro = similarity(edge_index_, node_deg, self.data)

    def plot_points(self, colors):
        self.model_gnn.eval()  # 将 GNN 模型设置为评估模式
        z = self.model_gnn(self.data)  # 获取节点嵌入
        z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())   # 将嵌入降维到二维
        y = self.data.y.cpu().numpy()  # 获取节点的标签

        plt.figure(figsize=(8, 8))
        for i in range(self.nclass):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])  # 绘制不同类别的点
        plt.axis('off')
        plt.show()

    def train(self):

        for fold in range(self.args.folds):
            set_random_seeds(fold)
            self.train_mask, self.val_mask, self.test_mask = masking(fold, self.data, self.args.label_rate)
            self._init_dataset()
            self._init_model()
            adj_csr = self.getadj(self.data, self.device)
            edge_index2, x_2 = create_diffusion_graph(adj_csr, self.data.x, self.device)
            #edge_index2, x_2 = create_pseudo_label_graph(self.data.x, self.device)
            #edge_index2, x_2 = label_smoothing_graph(adj_csr, self.data.x, self.device)
            #edge_index2, x_2 = create_attention_graph(adj_csr, self.data.x, self.device)
            #edge_index2, x_2 = create_diffusion_graph_with_lpa(adj_csr, self.data.x, self.device)
            self.Sampler = Sampler(self.args, self.data, self.labels, self.running_train_mask)
            self._init_model()
            for epoch in range(1, self.args.epochs+1):
                self.model_gnn.train()
                self.model_gnnt.train()
                # forward
                self.optimizer.zero_grad()

                positive = self.transform2(self.data)
                label_matrix, support_index, self.batch_size = self.Sampler.sample()
                diffg = Data(x=x_2, edge_index=edge_index2)
                with torch.no_grad():
                    teacher_original = self.model_gnnt(diffg)
                original_support_rep=teacher_original[support_index]
                pos_rep = self.model_gnn(positive)
                pos_support_rep = pos_rep[support_index]
                sup_loss = 0.
                logits, _ = self.model_gnn.cls(positive)
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                # sup_loss += F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
                out2 = logits
                edge_index1, x_1, new_index, new_prediction = other_transform(self.args, self.data, self.device,
                                                                              self.predict_lbl_pro, self.degree_sim,
                                                                              self.weights, out2)
                # anchor = Data(x=self.data.x, edge_index=edge_index1)
                anchor = Data(x=x_1, edge_index=edge_index1)
                anchor_rep = self.model_gnn(anchor)
                anchor_support_rep = anchor_rep[support_index]
                logits, _ = self.model_gnn.cls(anchor)
                # sup_loss += F.cross_entropy(logits[self.train_mask], self.labels[self.train_mask])
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                sup_loss /= 2
                out = logits

                consistency_loss = self.model_gnn.loss(anchor_rep, pos_rep, anchor_support_rep, pos_support_rep, label_matrix, self.data.y, self.train_mask)

                '''sup_loss = 0.
                logits, _ = self.model.cls(anchor)
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                logits, _ = self.model.cls(positive)
                sup_loss += F.cross_entropy(logits[self.running_train_mask], self.labels[self.running_train_mask])
                sup_loss /= 2'''

                # unsupervised loss
                unsup_loss = 2 - 2* F.cosine_similarity(anchor_rep, pos_rep, dim=-1).mean()
                P = torch.rand(anchor_support_rep.shape[0], anchor_support_rep.shape[1]).cuda()
                Ones = torch.ones_like(P)
                total_test = P * pos_support_rep.detach() + (Ones - P) * anchor_support_rep.detach()
                loss3 = self.similar_loss(original_support_rep, total_test).mean()

                loss4=self.model_gnn.loss_infomax(teacher_original,pos_rep)
                loss5=self.model_gnn.loss_infomax(teacher_original,anchor_rep)
                loss6=0.5*loss4+0.5*loss5
                loss = sup_loss + self.args.lam*consistency_loss + self.args.lam2 * unsup_loss+self.args.lam3*loss3+self.args.lam4*loss6

                loss.backward()
                self.optimizer.step()
                self.teacher_update_ma()
                st = '[Fold : {}][Epoch {}/{}] Consistency_Loss: {:.4f} | Sup_loss : {:.4f} | Unsup_loss : {:.4f} | Total_loss : {:.4f}'.format(
                        fold+1, epoch, self.args.epochs, consistency_loss.item(), sup_loss.item(), unsup_loss.item(), loss.item())

                # evaluation
                self.evaluate(self.data, st)

                if self.cnt == self.args.patience:
                    print("early stopping!")
                    break

            self.save_results(fold)
            
        self.summary()
        colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']
        #self.plot_points(colors)


class TGCL(nn.Module):
    def __init__(self, encoder, classifier, unique_labels, tau=0.1, thres=0.9, device=0,flag=1):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

        self.tau = tau
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)
        self.num_unique_labels = len(unique_labels)

        self.device = device

        self.reset_parameters()
        self.flag=flag
        
    def forward(self, x):
        if self.flag:
            rep = self.encoder(x)
        else:
            rep = self.encoder(x)
        return rep

    def snn(self, query, supports, labels):
        query = F.normalize(query)
        supports = F.normalize(supports)

        return self.softmax(query @ supports.T / self.tau) @ labels

    def loss(self, anchor, pos, anchor_supports, pos_supports, labels, gt_labels, train_mask):
        
        with torch.no_grad():
            gt_labels = gt_labels[train_mask].unsqueeze(-1)
            matrix = torch.zeros(train_mask.sum().item(), self.num_unique_labels).to(self.device)
            gt_matrix = matrix.scatter_(1, gt_labels, 1)

        probs1 = self.snn(anchor, anchor_supports, labels)
        with torch.no_grad():
            targets1 = self.snn(pos, pos_supports, labels)
            values, _ = targets1.max(dim=1)
            boolean = torch.logical_or(values>self.thres, train_mask)            
            indices1 = torch.arange(len(targets1))[boolean]
            targets1[targets1 < 1e-4] *= 0
            targets1[train_mask] = gt_matrix            
            targets1 = targets1[indices1]

        probs1 = probs1[indices1]
        loss = torch.mean(torch.sum(torch.log(probs1**(-targets1)), dim=1))

        return loss

    def loss_infomax(self, x, x_cl):
        T = 0.2
        with torch.no_grad():
            batch_size, _ = x.size()
            x_abs = x.norm(dim=1)  #|x|
            x_cl_abs = x_cl.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
            sim_matrix = torch.exp(sim_matrix / T)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss
    def cls(self, x):
        if self.flag:
            out = self.encoder(x)
        else:
            out=self.encoder(x)
        return self.classifier(out)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)

