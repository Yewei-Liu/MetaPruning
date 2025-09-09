from torch_geometric.nn import MessagePassing
from torch import Tensor
from torch_geometric.nn import PNAConv
import torch.nn as nn
import torch

class PNAConv(MessagePassing):
    def __init__(self, hiddim: int, aggr = ["mean", "std", "max", "min"], flow = "source_to_target", node_dim = -2):
        super().__init__(aggr, flow=flow, node_dim=node_dim, )
        self.lin1 = nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))
        self.lin2 = nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))
        self.lin3 = nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))
        self.lin4 = nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))
        self.downlin = nn.Linear(4*hiddim, hiddim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        ret_node = self.downlin(self.propagate(edge_index, x=self.lin1(x), z=self.lin2(x), edge_attr=edge_attr))
        ret_edge = self.lin3(x)[edge_index[0]] * self.lin4(x)[edge_index[1]] * edge_attr  
        return ret_node, ret_edge

    def message(self, z_i, x_j, edge_attr):
        return z_i * x_j * edge_attr

class MyGNN(nn.Module):
    def __init__(self, num_layer: int, hiddim: int, in_node_dim: int, in_edge_dim: int, node_res_ratio: float=0.01, edge_res_ratio: float=0.01):
        super().__init__()
        self.node_res_ratio = node_res_ratio
        self.edge_res_ratio = edge_res_ratio
        self.nodeEnc = nn.Linear(in_node_dim, hiddim)
        

        self.convs = nn.ModuleList([PNAConv(hiddim) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(hiddim, elementwise_affine=False)

        self.num_layer = num_layer
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))for _ in range(num_layer)])
        self.edgeEnc = nn.Linear(in_edge_dim, hiddim)
        self.register_buffer("edgeInverter", torch.concat((torch.ones((hiddim//2,)), -torch.ones(((hiddim+1)//2,))), dim=0))
        self.edgeDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, in_edge_dim))
        self.nodeDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, in_node_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        hidden = self.nodeEnc(x)
        in_edge_attr = edge_attr
        edge_attr = self.edgeEnc(edge_attr)
        for i in range(self.num_layer):
            ret_node1, ret_edge1 = self.convs[i].forward(self.norm(hidden), edge_index, edge_attr) 
            ret_node2, ret_edge2 = self.convs[i].forward(self.norm(hidden), edge_index[[1, 0]], self.edgeInverter * edge_attr)
            hidden = hidden + ret_node1 + ret_node2
            edge_attr = edge_attr + ret_edge1 + self.edgeInverter * ret_edge2
        # print(hidden.shape, edge_attr.shape, self.edgeDec)
        edgeout = self.edge_res_ratio * self.edgeDec(edge_attr)
        edge_pred = in_edge_attr + edgeout
        node_pred = x + self.node_res_ratio * self.nodeDec(hidden)
        return node_pred, edge_pred


class MyGNN_resnet50(nn.Module):
    def __init__(self, num_layer: int, hiddim: int, in_node_dim: int, node_res_ratio: float=0.01, edge_res_ratio: float=0.01):
        super().__init__()
        self.node_res_ratio = node_res_ratio
        self.edge_res_ratio = edge_res_ratio
        self.nodeEnc = nn.Linear(in_node_dim, hiddim)
        

        self.convs = nn.ModuleList([PNAConv(hiddim) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(hiddim, elementwise_affine=False)

        self.num_layer = num_layer
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))for _ in range(num_layer)])
        self.edgeEnc_49 = nn.Linear(49, hiddim)
        self.edgeEnc_9 = nn.Linear(9, hiddim)
        self.edgeEnc_1 = nn.Linear(1, hiddim)
        self.register_buffer("edgeInverter", torch.concat((torch.ones((hiddim//2,)), -torch.ones(((hiddim+1)//2,))), dim=0))
        self.edgeDec_49 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 49))
        self.edgeDec_9 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 9))
        self.edgeDec_1 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 1))
        self.edgeDec = [self.edgeDec_49, self.edgeDec_9, self.edgeDec_1]
        self.nodeDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, in_node_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr_list: Tensor):
        hidden = self.nodeEnc(x)
        edge_num = [len(i) for i in edge_attr_list]
        edge_attr = torch.concatenate([self.edgeEnc_49(edge_attr_list[0]), self.edgeEnc_9(edge_attr_list[1]), self.edgeEnc_1(edge_attr_list[2])], dim=0)
        for i in range(self.num_layer):
            ret_node1, ret_edge1 = self.convs[i].forward(self.norm(hidden), edge_index, edge_attr) 
            ret_node2, ret_edge2 = self.convs[i].forward(self.norm(hidden), edge_index[[1, 0]], self.edgeInverter * edge_attr)
            hidden = hidden + ret_node1 + ret_node2
            edge_attr = edge_attr + ret_edge1 + self.edgeInverter * ret_edge2
        idx = 0
        edge_pred_list = []
        for i, num in enumerate(edge_num):
            edge_pred_list.append(self.edgeDec[i](edge_attr[idx:idx+num]))
            idx += num
        edge_pred_list = [edge_attr_list[i] + self.edge_res_ratio * edge_pred_list[i] for i in range(len(edge_attr_list))]
        node_pred = x + self.node_res_ratio * self.nodeDec(hidden)
        return node_pred, edge_pred_list
    
    
class MyGNN_ViT(nn.Module):
    def __init__(self, num_layer: int, hiddim: int, in_node_dim: int, node_res_ratio: float=0.01, edge_res_ratio: float=0.01):
        super().__init__()
        self.node_res_ratio = node_res_ratio
        self.edge_res_ratio = edge_res_ratio
        self.nodeEnc_0 = nn.Linear(in_node_dim, hiddim)
        self.nodeEnc_1 = nn.Linear(in_node_dim, hiddim)
        

        self.convs = nn.ModuleList([PNAConv(hiddim) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(hiddim, elementwise_affine=False)

        self.num_layer = num_layer
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True))for _ in range(num_layer)])
        self.edgeEnc_49 = nn.Linear(49, hiddim)
        self.edgeEnc_9 = nn.Linear(9, hiddim)
        self.edgeEnc_1 = nn.Linear(1, hiddim)
        self.register_buffer("edgeInverter", torch.concat((torch.ones((hiddim//2,)), -torch.ones(((hiddim+1)//2,))), dim=0))
        self.edgeDec_49 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 49))
        self.edgeDec_9 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 9))
        self.edgeDec_1 = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, 1))
        self.edgeDec = [self.edgeDec_49, self.edgeDec_9, self.edgeDec_1]
        self.nodeDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, hiddim), nn.SiLU(inplace=True), nn.Linear(hiddim, in_node_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr_list: Tensor):
        hidden = self.nodeEnc(x)
        edge_num = [len(i) for i in edge_attr_list]
        edge_attr = torch.concatenate([self.edgeEnc_49(edge_attr_list[0]), self.edgeEnc_9(edge_attr_list[1]), self.edgeEnc_1(edge_attr_list[2])], dim=0)
        for i in range(self.num_layer):
            ret_node1, ret_edge1 = self.convs[i].forward(self.norm(hidden), edge_index, edge_attr) 
            ret_node2, ret_edge2 = self.convs[i].forward(self.norm(hidden), edge_index[[1, 0]], self.edgeInverter * edge_attr)
            hidden = hidden + ret_node1 + ret_node2
            edge_attr = edge_attr + ret_edge1 + self.edgeInverter * ret_edge2
        idx = 0
        edge_pred_list = []
        for i, num in enumerate(edge_num):
            edge_pred_list.append(self.edgeDec[i](edge_attr[idx:idx+num]))
            idx += num
        edge_pred_list = [edge_attr_list[i] + self.edge_res_ratio * edge_pred_list[i] for i in range(len(edge_attr_list))]
        node_pred = x + self.node_res_ratio * self.nodeDec(hidden)
        return node_pred, edge_pred_list