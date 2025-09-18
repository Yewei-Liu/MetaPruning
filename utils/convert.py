import torch
import torch.nn.functional as F
from copy import deepcopy
from generate_dataset.resnet_family import MyResNet, resnet56
from generate_dataset.VGG_family import MyVGG
from generate_dataset.resnet_deep_family import MyResNetDeep, myresnet50
from generate_dataset.vision_transformer import vit_b_16, my_vit_b_16



def state_dict_to_graph(model_name, state_dict, device=None):
    if model_name == 'resnet56':
        return resnet56_state_dict_to_graph(state_dict, device)
    elif model_name == 'resnet110':
        return resnet110_state_dict_to_graph(state_dict, device)
    elif model_name == 'VGG19':
        return VGG19_state_dict_to_graph(state_dict, device)
    elif model_name == 'resnet50':
        return resnet50_state_dict_to_graph(state_dict, device)
    elif model_name == 'vit_b_16':
        return ViT_B_16_state_dict_to_graph(state_dict, device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def graph_to_state_dict(model_name, state_dict, node_index, node_features, edge_index, edge_features, device=None):
    if model_name == 'resnet56':
        return resnet56_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'resnet110':
        return resnet110_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'VGG19':
        return VGG19_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'resnet50':
        return resnet50_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    elif model_name == 'vit_b_16':
        return ViT_B_16_graph_to_state_dict(state_dict, node_index, node_features, edge_index, edge_features, device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

def state_dict_to_model(model_name, state_dict, device='cuda'):
    if model_name == 'resnet56':
        return resnet56_state_dict_to_model(state_dict, device)
    elif model_name == 'resnet110':
        return resnet110_state_dict_to_model(state_dict, device)
    elif model_name == 'VGG19':
        return VGG19_state_dict_to_model(state_dict, device)
    elif model_name == 'resnet50':
        return resnet50_state_dict_to_model(state_dict, device)
    elif model_name == 'vit_b_16':
        return ViT_B_16_state_dict_to_model(state_dict, device)
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
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-4], node_index[-3])))])
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
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                tmp = val.reshape(val.shape[0], val.shape[1], 1, 1)
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
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
            else:
                edge_index = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
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
                    edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-4], node_index[-3])))])
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
    res = MyResNet(56, node_num).eval().to(device)
    res.load_state_dict(state_dict)
    return res

def resnet110_state_dict_to_graph(state_dict, device=None):
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
        if t >= 234 and t < 240 or t >= 456 and t < 462:
            if t % 6 == 0:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-4], node_index[-3])))])
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
        elif t >= 666:
            if t == 666:
                node_index.append(node_index[-1] + len(val))
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                tmp = val.reshape(val.shape[0], val.shape[1], 1, 1)
                edge_features = torch.concatenate([edge_features, F.pad(tmp, pad=[1, 1, 1, 1]).reshape(-1, 9)])
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), (len(val), 1))
            elif t == 667:
                new_node_features[:, 1] = val
                node_features = torch.concatenate([node_features, new_node_features])
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
            if edge_index is not None:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
            else:
                edge_index = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
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
                    edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-4], node_index[-3])))])
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

def resnet110_graph_to_state_dict(
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
        if t >= 234 and t < 240 or t >= 456 and t < 462:
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

def resnet110_state_dict_to_model(state_dict, device):
    node_index = [0, 3]
    t = 0
    for key, val in state_dict.items():
        if t >= 234 and t < 240 or t >= 456 and t < 462:
            pass
        elif t >= 666:
            if t == 666:
                node_index.append(node_index[-1] + len(val))
            else:
                break
        elif t % 6 == 0:
            node_index.append(node_index[-1] + len(val))
        t += 1
    node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
    res = MyResNet(110, node_num).eval().to(device)
    res.load_state_dict(state_dict)
    return res


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
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-5], node_index[-4])))])
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
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                tmp = val.reshape(-1, 1)
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
                    edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                else:
                    edge_index_1 = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
                if edge_features_1 is not None:
                    edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
                else:
                    edge_features_1 = val.reshape(-1, 1)
            elif val.shape[-1] == 3:
                if edge_index_9 is not None:
                    edge_index_9 = torch.concatenate([edge_index_9, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                else:
                    edge_index_9 = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
                if edge_features_9 is not None:
                    edge_features_9 = torch.concatenate([edge_features_9, val.reshape(-1, 9)])
                else:
                    edge_features_9 = val.reshape(-1, 9)
            else: # 7
                if edge_index_49 is not None:
                    edge_index_49 = torch.concatenate([edge_index_49, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                else:
                    edge_index_49 = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
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
            edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
            tmp = val.reshape(val.shape[0], val.shape[1], 1, 1)
            edge_features = torch.concatenate([edge_features, F.pad(tmp, pad=[1, 1, 1, 1]).reshape(-1, 9)])
            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]), (len(val), 1))
        elif t == 113:
            new_node_features[:, 4] = val
            node_features = torch.concatenate([node_features, new_node_features])
        elif t % 7 == 0:
            node_index.append(node_index[-1] + len(val))
            if edge_index is not None:
                edge_index = torch.concatenate([edge_index, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
            else:
                edge_index = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
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

def ViT_B_16_state_dict_to_graph(state_dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_index = [0, 3]
    node_features = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # Node feature matrix with shape [num_nodes, num_node_features]. 
    # 0: [layer norm weight, layer norm bias, linear bias, query bias, key bias, value bias]

    edge_index_1 = None
    edge_index_3 = None
    edge_index_256 = None 
    # Graph connectivity in COO format with shape [2, num_edges]. 

    edge_features_1 = None 
    edge_features_3 = None
    edge_features_256 = None 
    # Edge feature matrix with shape [num_edges, num_edge_features].
    # [16 * 16 weight]   

    state_dict = deepcopy(state_dict)
    node_index.append(3 + state_dict['conv_proj.weight'].shape[0])
    edge_index_256 = torch.cartesian_prod(torch.tensor(range(3, node_index[-1])), torch.tensor(range(0, 3)))
    edge_features_256 = state_dict['conv_proj.weight'].reshape(-1, 256).to(device)
    new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (node_index[-1] - node_index[-2], 1))
    new_node_features[:, 2] = state_dict['conv_proj.bias']
    t = 0
    for key, val in state_dict.items():
        t += 1
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if t <= 4:
            continue
        elif t <= 148:
            if t % 12 == 5:
                new_node_features[:, 0] = val
            elif t % 12 == 6:
                new_node_features[:, 1] = val
                node_features = torch.concatenate([node_features, new_node_features])
            elif t % 12 == 7:
                tmp_dim = len(val) // 3
                node_index.append(node_index[-1] + tmp_dim)
                if edge_index_3 is not None:
                    edge_index_3 = torch.concatenate([edge_index_3, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                else:
                    edge_index_3 = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
                if edge_features_3 is not None:
                    edge_features_3 = torch.concatenate([edge_features_3, torch.concatenate([val[:tmp_dim, :].reshape(-1, 1), val[tmp_dim: 2 * tmp_dim, :].reshape(-1, 1), val[2 * tmp_dim:, :].reshape(-1, 1)], dim=1)])
                else:
                    edge_features_3 = torch.concatenate([val[:tmp_dim, :].reshape(-1, 1), val[tmp_dim: 2 * tmp_dim, :].reshape(-1, 1), val[2 * tmp_dim:, :].reshape(-1, 1)], dim=1)
            elif t % 12 == 8:
                new_node_features = torch.zeros((len(val) // 3, 6))
                new_node_features[:, 3] = val[0::3]
                new_node_features[:, 4] = val[1::3]
                new_node_features[:, 5] = val[2::3]
                node_features = torch.concatenate([node_features, new_node_features]) if node_features is not None else new_node_features
            elif t % 12 == 9:
                node_index.append(node_index[-1] + val.shape[0])
                if edge_index_1 is not None:
                    edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                else:
                    edge_index_1 = torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))
                if edge_features_1 is not None:
                    edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
                else:
                    edge_features_1 = val.reshape(-1, 1)
                edge_index_1 = torch.concatenate([edge_index_1, torch.tensor([[i, i - node_index[-2] + node_index[-4]] for i in range(node_index[-2], node_index[-1])])])    
                edge_features_1 = torch.concatenate([edge_features_1, torch.ones((node_index[-1] - node_index[-2]), 1).to(edge_features_1.device)])
            elif t % 12 == 10:
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (len(val), 1))
                new_node_features[:, 2] = val
            elif t % 12 == 11:
                new_node_features[:, 0] = val
            elif t % 12 == 0:
                new_node_features[:, 1] = val
                node_features = torch.concatenate([node_features, new_node_features])
            elif t % 12 == 1:
                node_index.append(node_index[-1] + val.shape[0])
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
            elif t % 12 == 2:
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (len(val), 1))
                new_node_features[:, 2] = val
                node_features = torch.concatenate([node_features, new_node_features])
            elif t % 12 == 3:
                node_index.append(node_index[-1] + val.shape[0])
                edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
                edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
                edge_index_1 = torch.concatenate([edge_index_1, torch.tensor([[i, i - node_index[-2] + node_index[-4]] for i in range(node_index[-2], node_index[-1])])])    
                edge_features_1 = torch.concatenate([edge_features_1, torch.ones((node_index[-1] - node_index[-2]), 1).to(edge_features_1.device)])
            elif t % 12 == 4:
                new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (len(val), 1))
                new_node_features[:, 2] = val
        elif t == 149:
            new_node_features[:, 0] = val
        elif t == 150:
            new_node_features[:, 1] = val
            node_features = torch.concatenate([node_features, new_node_features])
        elif t == 151:
            node_index.append(node_index[-1] + val.shape[0])
            edge_index_1 = torch.concatenate([edge_index_1, torch.cartesian_prod(torch.tensor(range(node_index[-2], node_index[-1])), torch.tensor(range(node_index[-3], node_index[-2])))])
            edge_features_1 = torch.concatenate([edge_features_1, val.reshape(-1, 1)])
        elif t == 152:
            new_node_features = torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (len(val), 1))
            new_node_features[:, 2] = val
            node_features = torch.concatenate([node_features, new_node_features])
        
    edge_index = torch.concatenate([edge_index_256, edge_index_3, edge_index_1], dim=0).T
    edge_features_list = [edge_features_256.to(device), edge_features_3.to(device), edge_features_1.to(device)]
    return node_index, node_features.to(device), edge_index.to(device), edge_features_list

def ViT_B_16_graph_to_state_dict(
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
    edge_features_256, edge_features_3, edge_features_1 = edge_features
    
    state_dict['conv_proj.weight'] = edge_features_256.reshape(node_num[1], node_num[0], 16, 16)
    state_dict['conv_proj.bias'] = node_features[node_index[1]: node_index[2]][:, 2]
    node_idx = 1
    edge_idx_1 = 0
    edge_idx_3 = 0
    
    t = 0
    for key, val in state_dict.items():
        t += 1
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if t <= 4:
            continue
        elif t <= 148:
            if t % 12 == 5:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
            elif t % 12 == 6:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
                node_idx += 1
            elif t % 12 == 7:
                tmp_dim = val.shape[1] * val.shape[0] // 3
                Q = edge_features_3[edge_idx_3: edge_idx_3 + tmp_dim, 0].reshape(-1, val.shape[1])
                K = edge_features_3[edge_idx_3: edge_idx_3 + tmp_dim, 1].reshape(-1, val.shape[1])
                V = edge_features_3[edge_idx_3: edge_idx_3 + tmp_dim, 2].reshape(-1, val.shape[1])
                state_dict[key] = torch.concatenate([Q, K, V], dim=0)
                edge_idx_3 += tmp_dim
            elif t % 12 == 8:
                state_dict[key][0::3] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 3]
                state_dict[key][1::3] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 4]
                state_dict[key][2::3] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 5]
                node_idx += 1
            elif t % 12 == 9:
                tmp_dim = val.shape[0] * val.shape[1]
                state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + tmp_dim].reshape(val.shape)
                edge_idx_1 += tmp_dim
                edge_idx_1 += node_num[node_idx]
            elif t % 12 == 10:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 2]
            elif t % 12 == 11:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
            elif t % 12 == 0:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
                node_idx += 1
            elif t % 12 == 1:
                tmp_dim = val.shape[0] * val.shape[1]
                state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + tmp_dim].reshape(val.shape)
                edge_idx_1 += tmp_dim
            elif t % 12 == 2:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 2]
                node_idx += 1
            elif t % 12 == 3:
                tmp_dim = val.shape[0] * val.shape[1]
                state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + tmp_dim].reshape(val.shape)
                edge_idx_1 += tmp_dim
                edge_idx_1 += node_num[node_idx]
            elif t % 12 == 4:
                state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 2]
        elif t == 149:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 0]
        elif t == 150:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 1]
            node_idx += 1
        elif t == 151:
            tmp_dim = val.shape[0] * val.shape[1]
            state_dict[key] = edge_features_1[edge_idx_1: edge_idx_1 + tmp_dim].reshape(val.shape)
            edge_idx_1 += tmp_dim
        elif t == 152:
            state_dict[key] = node_features[node_index[node_idx]: node_index[node_idx + 1]][:, 2]
            node_idx += 1
        
    for key, val in state_dict.items():
        state_dict[key] = val.to(device)
    return state_dict

def ViT_B_16_state_dict_to_model(state_dict, device):
    node_index = [0, 3]
    t = 0
    for key, val in state_dict.items():
        t += 1
        if t == 2:
            node_index.append(node_index[-1] + val.shape[0])
        elif t <= 4:
            continue
        elif t <= 148:
            if t % 12 == 7:
                tmp_dim = len(val) // 3
                node_index.append(node_index[-1] + tmp_dim)
            elif t % 12 == 9:
                node_index.append(node_index[-1] + val.shape[0])
            elif t % 12 == 1:
                node_index.append(node_index[-1] + val.shape[0])
            elif t % 12 == 3:
                node_index.append(node_index[-1] + val.shape[0])
        elif t == 151:
            node_index.append(node_index[-1] + val.shape[0])
    node_num = [node_index[i + 1] - node_index[i] for i in range(len(node_index) - 1)]
    vit = my_vit_b_16(node_num).eval().to(device)
    vit.load_state_dict(state_dict)
    return vit

if __name__ == "__main__":
    model = vit_b_16()
    origin_state_dict = model.state_dict()
    # node_index, node_features, edge_index, edge_features_list = ViT_B_16_state_dict_to_graph(origin_state_dict, device='cpu')
    new_model = ViT_B_16_state_dict_to_model(origin_state_dict, device='cpu')
    for key in origin_state_dict.keys():
        if not torch.equal(origin_state_dict[key], new_model.state_dict()[key]):
            print(f"Key {key} not equal!")
    print("Done!")
