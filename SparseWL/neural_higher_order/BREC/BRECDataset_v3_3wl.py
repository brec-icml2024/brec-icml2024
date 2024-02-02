import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm
import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import preprocessing as pre

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def add_x(data):
    data.x = torch.ones([data.num_nodes, 1]).to(torch.float)
    return data


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3_3wl_nodenum.txt"]

    @property
    def processed_file_names(self):
        return ["brec_v3_3wl.pt"]

    def process(self):
        # data_list = np.load(self.raw_paths[0], allow_pickle=True)
        # data_list = [graph6_to_pyg(data) for data in data_list]

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        node_num = np.loadtxt(self.raw_paths[0], dtype=int)
        data_list = []
        matrices = pre.get_all_matrices_brec("brec_v3_3wl")
        node_labels = pre.get_all_node_labels_brec(False, False)
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.num_nodes = node_num[i] ** 2
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            one_hot = np.eye(3)[node_labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
