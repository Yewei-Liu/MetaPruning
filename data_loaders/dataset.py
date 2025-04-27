import torch
import torch.nn.functional as F
from utils.convert import state_dict_to_graph, state_dict_to_model
from collections import OrderedDict

class DatasetModel(torch.utils.data.Dataset):
    def __init__(self, model_name, dataset):
        self.dataset = dataset
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # state_dict to graph
        data = self.dataset[idx]
        train_acc = data.pop("train_acc")
        train_loss = data.pop("train_loss")
        val_acc = data.pop("val_acc")
        val_loss = data.pop("val_loss")
        current_speed_up = data.pop('current_speed_up')
        info = {"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "current_speed_up": current_speed_up}
        state_dict = OrderedDict()
        for key, val in data.items():
            state_dict[key] = torch.tensor(val)
        node_index, node_features, edge_index, edge_features = state_dict_to_graph(self.model_name, state_dict)
        model = state_dict_to_model(self.model_name, state_dict)
        return model, state_dict, info, node_index, node_features, edge_index, edge_features