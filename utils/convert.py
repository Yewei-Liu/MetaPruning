import torch
import torch.nn.functional as F
from torch_geometric.utils.undirected import to_undirected
from copy import deepcopy
from generate_dataset.resnet_family import MyResNet
from generate_dataset.VGG_family import MyVGG
from generate_dataset.resnet_deep_family import MyResNetDeep



def state_dict_to_graph(model_name, state_dict, device=None):
    if model_name == 'resnet56':
        return resnet56_state_dict_to_graph(state_dict, device)
    elif model_name == 'VGG19':
        return VGG19_state_dict_to_graph(state_dict, device)
    elif model_name == 'resnet50':
        return resnet50_state_dict_to_graph(state_dict, device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def graph_to_state_dict(model_name, state_dict, node_index, node_features, edge_index, edge_features, device=None):
    if model_name == 'resnet56':
        return resnet56_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'VGG19':
        return VGG19_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'resnet50':
        return resnet50_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def state_dict_to_model(model_name, state_dict, device='cuda'):
    if model_name == 'resnet56':
        return resnet56_state_dict_to_model(state_dict, device)
    elif model_name == 'VGG19':
        return VGG19_state_dict_to_model(state_dict, device)
    elif model_name == 'resnet50':
        return resnet50_state_dict_to_model(state_dict, device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def graph_to_model(model_name, state_dict, node_index, node_features, edge_index, edge_features, device=None):
    state_dict = graph_to_state_dict(model_name, state_dict, node_index, node_features, edge_index, edge_features, device=None)    
    return state_dict_to_model(model_name, state_dict, device)


def resnet56_state_dict_to_graph(state_dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_index = [0, 3]
    node_features = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    # Node feature matrix with shape [num_nodes, num_node_features]. 
    # [batchnorm: weight, bias, running mean, running var
    #  downsample batchnorm: weight, bias, running mean, running var],

    edge_index = None 
    # Graph connectivity in COO format with shape [2, num_edges]. 

    edge_features = None 
    # Edge feature matrix with shape [num_edges, num_edge_features].
    # [3 * 3 weight]   

    t = 0
    res = 1
    # down sample : 126 240 
    # end: 342
    new_node_features = None
    state_dict = deepcopy(state_dict)
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if t >= 126 and t < 132 or t >= 240 and t < 246:
            if t % 6 == 0:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-4], node_index[-3])), torch.tensor(range(node_index[-2], node_index[-1])))])
                new_edge_features = F.pad(val, pad=[1, 1, 1, 1]).reshape(-1, 9)
                edge_features = torch.concatenate([edge_features, new_edge_features])
            elif t % 6 == 1:
                node_features[node_index[-2]: node_index[-1]][:, 4] = val
            elif t % 6 == 2:
                node_features[node_index[-2]: node_index[-1]][:, 5] = val
            elif t % 6 == 3:
                node_features[node_index[-2]: node_index[-1]][:, 6] = val
            elif t % 6 == 4:
                node_features[node_index[-2]: node_index[-1]][:, 7] = val
        elif t >=  342:
            if t == 342:
                node_index.append(node_index[-1] + len(val))
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
                tmp = val.T
                tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1, 1)
                edge_features = torch.concatenate([edge_features, F.pad(tmp, pad=[1, 1, 1, 1]).reshape(-1, 9)])
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
            elif t == 343:
                new_node_features[:, 1] = val
                node_features = torch.concatenate([node_features, new_node_features])
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
            if edge_index is not None:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
            else:
                edge_index = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
            if edge_features is not None:
                edge_features = torch.concatenate([edge_features, val.reshape(-1, 9)])
            else:
                edge_features = val.reshape(-1, 9)
            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
            if t != 0:
                res ^= 1
                x = node_index[-3] - node_index[-4]
                y = node_index[-1] - node_index[-2]
                if res == 1 and x == y:
                    edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-4], node_index[-3])), torch.tensor(range(node_index[-2], node_index[-1])))])
                    new_edge_features = torch.zeros((x, y, 9)).to(edge_features.device)
                    for i in range(x):
                        new_edge_features[i][i][4] = 1.0
                    edge_features = torch.concatenate([edge_features, new_edge_features.reshape(-1, 9)])
        elif t % 6 == 1:
            new_node_features[:, 0] = val
        elif t % 6 == 2:
            new_node_features[:, 1] = val
        elif t % 6 == 3:
            new_node_features[:, 2] = val
        elif t % 6 == 4:
            new_node_features[:, 3] = val
        elif t % 6 == 5:
            node_features = torch.concatenate([node_features, new_node_features])
        t += 1
    # edge_index, edge_features = to_undirected(edge_index.T, edge_features, reduce="mean")
    return node_index, node_features.to(device), edge_index.T.to(device), edge_features.to(device)

def resnet56_graph_to_state_dict(
        origin_state_dict,
        node_index : list, 
        node_features : torch.tensor, 
        edge_index : torch.tensor, 
        edge_features : torch.tensor,
        device = None
        ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = deepcopy(origin_state_dict)
    node_num = [node_index[i] - node_index[i-1] for i in range(1, len(node_index))]
    state_dict['conv1.weight'] = edge_features[:node_num[0] * node_num[1]].reshape(node_num[1], node_num[0], 3, 3)
    state_dict['bn1.weight'] = node_features[node_index[1]: node_index[2]][:, 0]
    state_dict['bn1.bias'] = node_features[node_index[1]: node_index[2]][:, 1]
    t = 6
    res = 1
    edge_idx = node_num[0] * node_num[1]
    node_idx = 1
    for key in list(state_dict.keys())[6: -3]:
        if t >= 126 and t < 132 or t >= 240 and t < 246:
            if t % 6 == 0:
                state_dict[key] = edge_features[edge_idx - node_num[node_idx] * node_num[node_idx - 2]: edge_idx].reshape(node_num[node_idx], node_num[node_idx - 2], 3, 3)[:, :, 1:2, 1:2]
            elif t % 6 == 1:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 4]
            elif t % 6 == 2:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 5]
        elif t % 6 == 0:
            state_dict[key] = edge_features[edge_idx: edge_idx + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 3, 3)
            edge_idx += node_num[node_idx] * node_num[node_idx + 1]
            node_idx += 1
            res ^= 1
            if res == 1:
                edge_idx += node_num[node_idx - 2] * node_num[node_idx]
        elif t % 6 == 1:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
        elif t % 6 == 2:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
        t += 1
    edge_idx += node_num[-1] * node_num[-2]
    state_dict['fc.weight'] = edge_features[-node_num[-1] * node_num[-2]:].reshape(node_num[-2], node_num[-1], 3, 3)[:, :, 1, 1].T
    state_dict['fc.bias'] = node_features[node_index[-2]: node_index[-1]][:, 1]
    for key, val in state_dict.items():
        state_dict[key] = val.to(device)
    return state_dict

def resnet56_state_dict_to_model(state_dict, device):
    node_index = [0, 3]
    t = 0
    for key, val in state_dict.items():
        if t >= 126 and t < 132 or t >= 240 and t < 246:
            pass
        elif t >=  342:
            if t == 342:
                node_index.append(node_index[-1] + len(val))
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
        t += 1
    node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
    res = MyResNet(56, node_num, 10).eval().to(device)
    res.load_state_dict(state_dict)
    return res


# def resnet50_state_dict_to_graph(state_dict):
#     node_index = [0, 3]
#     node_features = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
#     # Node feature matrix with shape [num_nodes, num_node_features]. 
#     # [batchnorm: weight, bias, running mean, running var
#     #  downsample batchnorm: weight, bias, running mean, running var],

#     edge_index_1 = None
#     edge_index_9 = None
#     edge_index_49 = None 
#     # Graph connectivity in COO format with shape [2, num_edges]. 

#     edge_features_1 = None 
#     edge_features_9 = None
#     edge_features_49 = None 
#     # Edge feature matrix with shape [num_edges, num_edge_features].
#     # [7 * 7 weight]   

#     t = 0
#     res = 0
#     # down sample : 126 240 
#     # end: 342
#     new_node_features = None
#     state_dict = deepcopy(state_dict)
#     for key, val in state_dict.items():
#         if not isinstance(val, torch.Tensor):
#             val = torch.tensor(val)
#         if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
#             # Shortcut with weights
#             if t % 6 == 0:
#                 edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-5], node_index[-4])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                 new_edge_features = val.reshape(-1, 1)
#                 edge_features_1 = torch.concatenate([edge_features_1, new_edge_features])
#             elif t % 6 == 1:
#                 node_features[node_index[-2]: node_index[-1]][:, 4] = val
#             elif t % 6 == 2:
#                 node_features[node_index[-2]: node_index[-1]][:, 5] = val
#             elif t % 6 == 3:
#                 node_features[node_index[-2]: node_index[-1]][:, 6] = val
#             elif t % 6 == 4:
#                 node_features[node_index[-2]: node_index[-1]][:, 7] = val
#         elif t >=  318:
#             # End
#             if t == 318:
#                 node_index.append(node_index[-1] + len(val))
#                 edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                 tmp = val.T
#                 tmp = tmp.reshape(-1, 1)
#                 edge_features_1 = torch.concatenate([edge_features_1, tmp])
#                 new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
#             elif t == 319:
#                 new_node_features[:, 1] = val
#                 node_features = torch.concatenate([node_features, new_node_features])
#             else:
#                 break
#         elif t % 6 == 0:
#             node_index.append(node_index[-1] + len(val))
#             if val.shape[-1] == 1:
#                 if edge_index_1 is not None:
#                     edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                 else:
#                     edge_index_1 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
#                 if edge_features_1 is not None:
#                     edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
#                 else:
#                     edge_features_1 = val.reshape(-1, 1)
#             elif val.shape[-1] == 3:
#                 if edge_index_9 is not None:
#                     edge_index_9 = torch.concatenate([edge_index_9, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                 else:
#                     edge_index_9 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
#                 if edge_features_9 is not None:
#                     edge_features_9 = torch.concatenate([edge_features_9, val.reshape(-1, 9)])
#                 else:
#                     edge_features_9 = val.reshape(-1, 9)
#             else: # 7
#                 if edge_index_49 is not None:
#                     edge_index_49 = torch.concatenate([edge_index_49, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                 else:
#                     edge_index_49 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
#                 if edge_features_49 is not None:
#                     edge_features_49 = torch.concatenate([edge_features_49, val.reshape(-1, 49)])
#                 else:
#                     edge_features_49 = val.reshape(-1, 49)

#             new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
#             # short cut without weights
#             if t != 0:
#                 res = (res + 1) % 3
#                 if len(node_index) >= 5 and res == 0 and node_index[-4] - node_index[-5] == node_index[-1] - node_index[-2]:
#                     x = y = node_index[-4] - node_index[-5]
#                     edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-5], node_index[-4])), torch.tensor(range(node_index[-2], node_index[-1])))])
#                     new_edge_features = torch.zeros((x, y, 1)).to(edge_features_1.device)
#                     for i in range(x):
#                         new_edge_features[i][i][0] = 1.0
#                     edge_features_1 = torch.concatenate([edge_features_1, new_edge_features.reshape(-1, 1)])
#         elif t % 6 == 1:
#             new_node_features[:, 0] = val
#         elif t % 6 == 2:
#             new_node_features[:, 1] = val
#         elif t % 6 == 3:
#             new_node_features[:, 2] = val
#         elif t % 6 == 4:
#             new_node_features[:, 3] = val
#         elif t % 6 == 5:
#             node_features = torch.concatenate([node_features, new_node_features])
#         t += 1
#     edge_index = torch.concatenate([edge_index_49, edge_index_9, edge_index_1], dim=0).T
#     edge_features_list = [edge_features_49, edge_features_9, edge_features_1]
#     return node_index, node_features, edge_index, edge_features_list

# def resnet50_graph_to_state_dict(
#         origin_state_dict,
#         node_index : list, 
#         node_features : torch.tensor, 
#         edge_index : torch.tensor, 
#         edge_features_list : torch.tensor,
#         device = None
#         ):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     state_dict = deepcopy(origin_state_dict)
#     node_num = [node_index[i] - node_index[i-1] for i in range(1, len(node_index))]
#     edge_features_49, edge_features_9, edge_features_1 = edge_features_list
#     state_dict['conv1.weight'] = edge_features_49.reshape(node_num[1], node_num[0], 7, 7)
#     state_dict['bn1.weight'] = node_features[node_index[1]: node_index[2]][:, 0]
#     state_dict['bn1.bias'] = node_features[node_index[1]: node_index[2]][:, 1]
#     t = 6
#     res = 0
#     edge_idx_9 = 0
#     edge_idx_1 = 0
#     node_idx = 1
#     for key in list(state_dict.keys())[6: -3]:
#         if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
#             if t % 6 == 0:
#                 state_dict[key] = edge_features_1[edge_idx_1 - node_num[node_idx] * node_num[node_idx - 3]: edge_idx_1].reshape(node_num[node_idx], node_num[node_idx - 3], 1, 1)
#             elif t % 6 == 1:
#                 state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 4]
#             elif t % 6 == 2:
#                 state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 5]
#         elif t % 6 == 0:
#             res = (res + 1) % 3
#             if res == 1 or res == 0:
#                 state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 1, 1)
#                 edge_idx_1 += node_num[node_idx] * node_num[node_idx + 1]
#             else:
#                 state_dict[key] = edge_features_9[edge_idx_9: edge_idx_9 + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 3, 3)
#                 edge_idx_9 += node_num[node_idx] * node_num[node_idx + 1]
#             node_idx += 1
#             if res == 0:
#                 edge_idx_1 += node_num[node_idx - 3] * node_num[node_idx]
#         elif t % 6 == 1:
#             state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
#         elif t % 6 == 2:
#             state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
#         t += 1
#     edge_idx_1 += node_num[-1] * node_num[-2]
#     state_dict['fc.weight'] = edge_features_1[-node_num[-1] * node_num[-2]:].reshape(node_num[-2], node_num[-1]).T
#     state_dict['fc.bias'] = node_features[node_index[-2]: node_index[-1]][:, 1]
#     for key, val in state_dict.items():
#         state_dict[key] = val.to(device)
#     return state_dict

# def resnet50_state_dict_to_model(state_dict, device):
#     node_index = [0, 3]
#     t = 0
#     for key, val in state_dict.items():
#         if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
#             pass
#         elif t >=  318:
#             if t == 318:
#                 node_index.append(node_index[-1] + len(val))
#             else:
#                 break
#         elif t % 6 == 0:
#             node_index.append(node_index[-1] + len(val))
#         t += 1
#     node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
#     res = MyResNetDeep(node_num).eval().to(device)
#     res.load_state_dict(state_dict)
#     return res


def resnet50_state_dict_to_graph(state_dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_index = [0, 3]
    node_features = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    # Node feature matrix with shape [num_nodes, num_node_features]. 
    # [batchnorm: weight, bias, running mean, running var
    #  downsample batchnorm: weight, bias, running mean, running var],

    edge_index_1 = None
    edge_index_9 = None
    edge_index_49 = None 
    # Graph connectivity in COO format with shape [2, num_edges]. 

    edge_features_1 = None 
    edge_features_9 = None
    edge_features_49 = None 
    # Edge feature matrix with shape [num_edges, num_edge_features].
    # [7 * 7 weight]   

    t = 0
    res = 0
    # down sample : 126 240 
    # end: 342
    new_node_features = None
    state_dict = deepcopy(state_dict)
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
            # Shortcut with weights
            if t % 6 == 0:
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-5], node_index[-4])), torch.tensor(range(node_index[-2], node_index[-1])))])
                new_edge_features = val.reshape(-1, 1)
                edge_features_1 = torch.concatenate([edge_features_1, new_edge_features])
            elif t % 6 == 1:
                node_features[node_index[-2]: node_index[-1]][:, 4] = val
            elif t % 6 == 2:
                node_features[node_index[-2]: node_index[-1]][:, 5] = val
            elif t % 6 == 3:
                node_features[node_index[-2]: node_index[-1]][:, 6] = val
            elif t % 6 == 4:
                node_features[node_index[-2]: node_index[-1]][:, 7] = val
        elif t >=  318:
            # End
            if t == 318:
                node_index.append(node_index[-1] + len(val))
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
                tmp = val.T
                tmp = tmp.reshape(-1, 1)
                edge_features_1 = torch.concatenate([edge_features_1, tmp])
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
            elif t == 319:
                new_node_features[:, 1] = val
                node_features = torch.concatenate([node_features, new_node_features])
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
            if val.shape[-1] == 1:
                if edge_index_1 is not None:
                    edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
                else:
                    edge_index_1 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
                if edge_features_1 is not None:
                    edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
                else:
                    edge_features_1 = val.reshape(-1, 1)
            elif val.shape[-1] == 3:
                if edge_index_9 is not None:
                    edge_index_9 = torch.concatenate([edge_index_9, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
                else:
                    edge_index_9 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
                if edge_features_9 is not None:
                    edge_features_9 = torch.concatenate([edge_features_9, val.reshape(-1, 9)])
                else:
                    edge_features_9 = val.reshape(-1, 9)
            else: # 7
                if edge_index_49 is not None:
                    edge_index_49 = torch.concatenate([edge_index_49, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
                else:
                    edge_index_49 = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
                if edge_features_49 is not None:
                    edge_features_49 = torch.concatenate([edge_features_49, val.reshape(-1, 49)])
                else:
                    edge_features_49 = val.reshape(-1, 49)

            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
            # short cut without weights
            if t != 0:
                res = (res + 1) % 3
                if len(node_index) >= 5 and res == 0 and node_index[-4] - node_index[-5] == node_index[-1] - node_index[-2]:
                    x = y = node_index[-4] - node_index[-5]
                    edge_index_1 = torch.concatenate([edge_index_1, torch.tensor([[node_index[-5] + i, node_index[-2] + i] for i in range(x)])])
                    new_edge_features = torch.ones((x, 1)).to(edge_features_1.device)
                    edge_features_1 = torch.concatenate([edge_features_1, new_edge_features])
        elif t % 6 == 1:
            new_node_features[:, 0] = val
        elif t % 6 == 2:
            new_node_features[:, 1] = val
        elif t % 6 == 3:
            new_node_features[:, 2] = val
        elif t % 6 == 4:
            new_node_features[:, 3] = val
        elif t % 6 == 5:
            node_features = torch.concatenate([node_features, new_node_features])
        t += 1
    edge_index = torch.concatenate([edge_index_49, edge_index_9, edge_index_1], dim=0).T
    edge_features_list = [edge_features_49.to(device), edge_features_9.to(device), edge_features_1.to(device)]
    return node_index, node_features.to(device), edge_index.to(device), edge_features_list

def resnet50_graph_to_state_dict(
        origin_state_dict,
        node_index : list, 
        node_features : torch.tensor, 
        edge_index : torch.tensor, 
        edge_features_list : torch.tensor,
        device = None
        ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = deepcopy(origin_state_dict)
    node_num = [node_index[i] - node_index[i-1] for i in range(1, len(node_index))]
    edge_features_49, edge_features_9, edge_features_1 = edge_features_list
    state_dict['conv1.weight'] = edge_features_49.reshape(node_num[1], node_num[0], 7, 7)
    state_dict['bn1.weight'] = node_features[node_index[1]: node_index[2]][:, 0]
    state_dict['bn1.bias'] = node_features[node_index[1]: node_index[2]][:, 1]
    t = 6
    res = 0
    edge_idx_9 = 0
    edge_idx_1 = 0
    node_idx = 1
    for key in list(state_dict.keys())[6: -3]:
        if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
            if t % 6 == 0:
                edge_idx_1 = edge_idx_1 - node_num[node_idx] + node_num[node_idx - 3] * node_num[node_idx]
                state_dict[key] = edge_features_1[edge_idx_1 - node_num[node_idx] * node_num[node_idx - 3]: edge_idx_1].reshape(node_num[node_idx], node_num[node_idx - 3], 1, 1)
            elif t % 6 == 1:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 4]
            elif t % 6 == 2:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 5]
        elif t % 6 == 0:
            res = (res + 1) % 3
            if res == 1 or res == 0:
                state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 1, 1)
                edge_idx_1 += node_num[node_idx] * node_num[node_idx + 1]
            else:
                state_dict[key] = edge_features_9[edge_idx_9: edge_idx_9 + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 3, 3)
                edge_idx_9 += node_num[node_idx] * node_num[node_idx + 1]
            node_idx += 1
            if res == 0:
                edge_idx_1 += node_num[node_idx]
        elif t % 6 == 1:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
        elif t % 6 == 2:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
        t += 1
    edge_idx_1 += node_num[-1] * node_num[-2]
    state_dict['fc.weight'] = edge_features_1[-node_num[-1] * node_num[-2]:].reshape(node_num[-2], node_num[-1]).T
    state_dict['fc.bias'] = node_features[node_index[-2]: node_index[-1]][:, 1]
    for key, val in state_dict.items():
        state_dict[key] = val.to(device)
    return state_dict

def resnet50_state_dict_to_model(state_dict, device):
    node_index = [0, 3]
    t = 0
    for key, val in state_dict.items():
        if (t >= 24 and t < 30) or (t >= 84 and t < 90) or (t >= 162 and t < 168) or (t >= 276 and t < 282):
            pass
        elif t >=  318:
            if t == 318:
                node_index.append(node_index[-1] + len(val))
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
        t += 1
    node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
    res = MyResNetDeep(node_num).eval().to(device)
    res.load_state_dict(state_dict)
    return res



def VGG19_state_dict_to_graph(state_dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_index = [0, 3]
    node_features = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]])
    # Node feature matrix with shape [num_nodes, num_node_features]. 
    # [batchnorm: weight, bias, running mean, running var,   bias]

    edge_index = None 
    # Graph connectivity in COO format with shape [2, num_edges]. 

    edge_features = None 
    # Edge feature matrix with shape [num_edges, num_edge_features].
    # [3 * 3 weight]   

    t = 0
    # end: 342
    new_node_features = None
    state_dict = deepcopy(state_dict)
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if t == 112:
            node_index.append(node_index[-1] + len(val))
            edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
            tmp = val.T
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1, 1)
            edge_features = torch.concatenate([edge_features, F.pad(tmp, pad=[1, 1, 1, 1]).reshape(-1, 9)])
            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]), (len(val), 1))
        elif t == 113:
            new_node_features[:, 4] = val
            node_features = torch.concatenate([node_features, new_node_features])
        elif t % 7 == 0:
            node_index.append(node_index[-1] + len(val))
            if edge_index is not None:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))])
            else:
                edge_index = torch.cartesian_prod(torch.tensor(range(node_index[-3], node_index[-2])), torch.tensor(range(node_index[-2], node_index[-1])))
            if edge_features is not None:
                edge_features = torch.concatenate([edge_features, val.reshape(-1, 9)])
            else:
                edge_features = val.reshape(-1, 9)
            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]), (len(val), 1))
        elif t % 7 == 1:
            new_node_features[:, 4] = val
        elif t % 7 == 2:
            new_node_features[:, 0] = val
        elif t % 7 == 3:
            new_node_features[:, 1] = val
        elif t % 7 == 4:
            new_node_features[:, 2] = val
        elif t % 7 == 5:
            new_node_features[:, 3] = val
        elif t % 7 == 6:
            node_features = torch.concatenate([node_features, new_node_features])
        t += 1
    return node_index, node_features.to(device), edge_index.T.to(device), edge_features.to(device)


def VGG19_graph_to_state_dict(
        origin_state_dict,
        node_index : list, 
        node_features : torch.tensor, 
        edge_index : torch.tensor, 
        edge_features : torch.tensor,
        device = None
        ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = deepcopy(origin_state_dict)
    node_num = [node_index[i] - node_index[i-1] for i in range(1, len(node_index))]
    t = 0
    edge_idx = 0
    node_idx = 0
    for key in list(state_dict.keys())[:-2]:
        if t % 7 == 0:
            state_dict[key] = edge_features[edge_idx: edge_idx + node_num[node_idx] * node_num[node_idx + 1]].reshape(node_num[node_idx + 1], node_num[node_idx], 3, 3)
            edge_idx += node_num[node_idx] * node_num[node_idx + 1]
            node_idx += 1
        elif t % 7 == 1:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 4]
        elif t % 7 == 2:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
        elif t % 7 == 3:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
        t += 1
    edge_idx += node_num[-1] * node_num[-2]
    state_dict['classifier.weight'] = edge_features[-node_num[-1] * node_num[-2]:].reshape(node_num[-2], node_num[-1], 3, 3)[:, :, 1, 1].T
    state_dict['classifier.bias'] = node_features[node_index[-2]: node_index[-1]][:, 4]
    for key, val in state_dict.items():
        state_dict[key] = val.to(device)
    return state_dict

def VGG19_state_dict_to_model(state_dict, device):
    node_index = [0, 3]
    t = 0
    for key, val in state_dict.items():
        if t % 7 == 0:
            node_index.append(node_index[-1] + len(val))
        t += 1
    node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
    vgg = MyVGG(node_num).eval().to(device)
    vgg.load_state_dict(state_dict)
    return vgg

if __name__ == "__main__":
    pass