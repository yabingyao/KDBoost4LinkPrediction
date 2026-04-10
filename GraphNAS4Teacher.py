import autogl
import typing as _typ
import torch
import torch.nn.functional as F
from autogl.module.nas.space.base import BaseSpace,BaseAutoModel
from autogl.module.nas.space.operation import gnn_map
from autogl.module.nas.backend import *
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.feature import OneHotDegreeGenerator
from autogl.module.nas.space import GraphNasNodeClassificationSpace, SinglePathNodeClassificationSpace
from autogl.module.nas.algorithm import GraphNasRL, Darts
from autogl.module.nas.estimator import OneShotEstimator,TrainEstimator
from torch_geometric.datasets import Planetoid
from os import path as osp
import torch_geometric.transforms as T
import numpy as np
import torch.nn as nn


class SinglePath(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 128,
        layer_number: _typ.Optional[int] = 2,
        dropout: _typ.Optional[float] = 0.2,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = ['gcn', "gat_8", "gat_6", "gat_4", "gat_2", "sage", "cheb"]
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout

    def instantiate(
        self,
        hidden_dim: _typ.Optional[int] = None,
        layer_number: _typ.Optional[int] = None,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        ops: _typ.Tuple = None,
        dropout=None,
    ):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.dropout = dropout or self.dropout
        for layer in range(self.layer_number):
            setattr(
                self,
                f"op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        op(
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        if isinstance(op, type)
                        else gnn_map(
                            op,
                            self.input_dim if layer == 0 else self.hidden_dim,
                            self.output_dim
                            if layer == self.layer_number - 1
                            else self.hidden_dim,
                        )
                        for op in self.ops
                    ],
                ),
            )
        self._initialized = True
    # Link prediction
    def forward(self, data):
        ## Encoder
        x, edge_index = data.x, data.edge_index
        for layer in range(self.layer_number):
            op= getattr(self, f"op_{layer}")
            x = bk_gconv(op,data,x)
            if layer != self.layer_number - 1:
                 x = F.leaky_relu(x)
        ## Decoder
        # PubMed
        # src = x[edge_index[0][:19717]]
        # dst = x[edge_index[1][:19717]]
        # Cora
        src = x[edge_index[0][:data.x.shape[0]]]
        dst = x[edge_index[1][:data.x.shape[0]]]
        x = (src * dst)
        r = x.sum(dim=1)
        return x        
    def parse_model(self, selection, device) -> BaseAutoModel:
        return self.wrap().fix(selection)


torch.manual_seed(20230331)
torch.cuda.manual_seed_all(20230331)

dataset = Planetoid(root='mnt/lp_data/', name='Cora')
data = dataset[0]
label = data.y
input_dim = data.x.shape[-1]
num_classes = len(np.unique(label.numpy()))
space = SinglePath(input_dim=input_dim, output_dim=64, layer_number=2)
space.instantiate()
algo = Darts(num_epochs=2)
estimator = OneShotEstimator()
algo.search(space, dataset, estimator)


