import torch
import torch_geometric.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class GAT(torch.nn.Module):
    def __init__(self, config, dataset):
        super(GAT, self).__init__()
        self.config = config
        self.conv1 = nn.GATConv(
            dataset.num_node_features,
            config['model']['hidden_dim'],
            heads=config['model']['heads'],
            dropout=config['model']['dropout_rate']
        )
        self.conv2 = nn.GATConv(
            config['model']['hidden_dim'] * config['model']['heads'],
            dataset.num_classes,
            heads=1,
            concat=False,
            dropout=config['model']['dropout_rate']
        )
    
    def forward(self, data):
        x = data.x
        edge_index = add_self_loops(data.edge_index)[0]
        x = F.dropout(
            x, p=self.config['model']['dropout_rate'], training=self.training
        )
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(
            x, p=self.config['model']['dropout_rate'], training=self.training
        )
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)