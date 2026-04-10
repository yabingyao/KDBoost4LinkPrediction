# -*- coding: utf-8 -*-
# Converted from notebook: KDBoost-4(1).ipynb

# In[1]:
# -*- coding: utf-8 -*-
# Full script: Graph2Feat link prediction (GNN/MLP/KD+Projector) + AUC/AUPR
# Evaluation uses only AUC and AUPR on the validation and test sets

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Coauthor,Twitch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv
from torch_geometric.utils import negative_sampling
from torch.nn import Linear 

# Metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# -------------------------
# Device & random seed
# -------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(202312)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(202312)

# -------------------------
# Dataset & transforms
# -------------------------
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

# Select dataset (default: Cora)
dataset = Planetoid(root='/mnt/data/Cora', name='Cora', transform=transform)
# dataset = Planetoid(root='/mnt/data/CiteSeer', name='CiteSeer', transform=transform)
# dataset = Planetoid(root='/mnt/data/PubMed', name='PubMed', transform=transform)
# dataset = Amazon(root='/mnt/data/dataset', name='Computers', transform=transform)
# dataset = WikipediaNetwork(root='/mnt/data/dataset', name='chameleon', transform=transform)
# dataset = Coauthor(root='/mnt/data/dataset', name='CS', transform=transform)
# dataset = Coauthor(root='/mnt/data/dataset', name='Physics', transform=transform)
# dataset = WikipediaNetwork(root='/mnt/data/dataset', name='squirrel', transform=transform)
# dataset = Twitch(root='/mnt/data/twitch', name='ES', transform=transform)
# dataset = Actor(root='/mnt/data/dataset/Actor', transform=transform)
train_data, val_data, test_data = dataset[0]

# -------------------------
# Negative sampling (using the training graph)
# -------------------------
def negative_sample():
    # Sample the same number of negative edges as positive edges from the training set
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
        method='sparse'
    )
    # Merge positive and negative samples
    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    return edge_label, edge_label_index

# -------------------------
# Model: GNN encoder (default: GCN)
# -------------------------
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, backbone: str = "gcn"):
        super().__init__()
        self.backbone = backbone.lower()

        if self.backbone == "sage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif self.backbone == "gat":
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif self.backbone == "cheb":
            self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
            self.conv2 = ChebConv(hidden_channels, out_channels, K=2)
        else:  # GCN
            heads = 2
            # self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True)
            # self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            # self.conv2 = ChebConv(hidden_channels, out_channels, K=2)

        self.GNN_Encoded_Result = None  # Store the encoded representation

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        self.GNN_Encoded_Result = x
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

# -------------------------
# Model: MLP encoder
# -------------------------
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.MLP_Encoded_Result = None

    def encode(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        self.MLP_Encoded_Result = x
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_label_index):
        z = self.encode(x)
        return self.decode(z, edge_label_index)

# -------------------------
# Model: Linear projector (for KD)
# -------------------------
class Projector(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)
        self.MLP_Encoded_Result = None  # Add a member variable to store the encoded representation

    def forward(self, x):
        x = self.linear(x)
        # x = torch.relu(x)
        return x

# -------------------------
# Evaluation utilities
# -------------------------
def collect_scores_lp(model, data):
    """Collect (y_true, y_prob); prob = sigmoid(logits). Automatically supports both GNN and MLP."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, GNN):
            z = model.encode(data.x, data.edge_index)
            logits = model.decode(z, data.edge_label_index).view(-1)
        else:  # MLP
            z = model.encode(data.x)
            logits = model.decode(z, data.edge_label_index).view(-1)
        probs = logits.sigmoid().cpu().numpy()
        y_true = data.edge_label.cpu().numpy()
    model.train()
    return y_true, probs

def evaluate_auc_aupr(y_true, y_prob):
    """Return (AUC, AUPR)."""
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return auc, ap

# -------------------------
# Training: GNN
# -------------------------
def fit_GNN(model, epochs):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_val_auc = 0.0

    # Record the test metrics corresponding to the best validation AUC
    final_test_auc = 0.0
    final_test_ap  = 0.0

    start = time.perf_counter()
    model.train()
    print('GNN_TrainResult')
    print('*********************************************************')
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, train_data.edge_index, edge_label_index).view(-1)
        GNN_Representation = model.GNN_Encoded_Result  # Used for subsequent KD
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        y_val, p_val = collect_scores_lp(model, val_data)
        val_auc, val_ap = evaluate_auc_aupr(y_val, p_val)

        y_test, p_test = collect_scores_lp(model, test_data)
        test_auc, test_ap = evaluate_auc_aupr(y_test, p_test)

        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_ap  = test_ap

        # Uncomment for per-epoch logs
        # print(f'epoch {epoch:03d} loss {loss.item():.6f} | '
        #       f'val AUC {val_auc:.4f} AP {val_ap:.4f} | '
        #       f'test AUC {test_auc:.4f} AP {test_ap:.4f}')

    end = time.perf_counter()
    run_time = end - start
    print("Runtime:", run_time)
    print('GNN_TestResult')
    print('*********************************************************')
    print('final_test_auc: {:.4f} | final_test_ap : {:.4f} | best_val_auc: {:.4f}'.format(
              final_test_auc, final_test_ap, best_val_auc))
    return GNN_Representation

# -------------------------
# Training: MLP
# -------------------------
def fit_MLP(model, epochs):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_val_auc = 0.0

    final_test_auc = 0.0
    final_test_ap  = 0.0

    start = time.perf_counter()
    model.train()
    print('MLP_Train')
    print('*********************************************************')
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        y_val, p_val = collect_scores_lp(model, val_data)
        val_auc, val_ap = evaluate_auc_aupr(y_val, p_val)

        y_test, p_test = collect_scores_lp(model, test_data)
        test_auc, test_ap = evaluate_auc_aupr(y_test, p_test)

        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_ap  = test_ap

    end = time.perf_counter()
    run_time = end - start
    print("Runtime:", run_time)
    print('MLP_Test')
    print('*********************************************************')
    print('final_test_auc: {:.4f} | final_test_ap : {:.4f} | best_val_auc: {:.4f}'.format(
              final_test_auc, final_test_ap, best_val_auc))
    return {
        'final_test_auc': final_test_auc,
        'final_test_ap':  final_test_ap,
        'best_val_auc':   best_val_auc
    }

# -------------------------
# Training: Knowledge distillation (GNN -> MLP) + Projector (kept according to your original logic)
# -------------------------
def KnowledgeDistillation(model, P_model1, P_model2, P_model3, dataset, epochs, a, GNN_Representation):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_val_auc = 0.0

    final_test_auc = 0.0
    final_test_ap  = 0.0

    start = time.perf_counter()
    model.train()
    print('KD_Train')
    print('*********************************************************')
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        edge_label, edge_label_index = negative_sample()
        out = model(train_data.x, edge_label_index).view(-1)

        # Current MLP representation
        MLP_Representation = model.MLP_Encoded_Result  # [N, d]

        # Projector with three heads (consistent with your implementation; fixed a small typo here: the third projector correctly uses P_model3)
        P_MLP_Representation1 = P_model1.forward(MLP_Representation)
        P_MLP_Representation2 = P_model2.forward(MLP_Representation)
        P_MLP_Representation3 = P_model3.forward(MLP_Representation)
        P_MLP_Sum_Representation = P_MLP_Representation1 + P_MLP_Representation2 + P_MLP_Representation3
        P_MLP_Representation = P_MLP_Sum_Representation / 3.0  # If you want to use it in KD, replace the next line with it

        # KD loss (consistent with your original logic: BCE + KL)
        kd_loss = F.kl_div(
            MLP_Representation.softmax(dim=-1).log(),
            GNN_Representation.softmax(dim=-1),
            reduction='sum'
        )
        loss = (1 - a) * criterion(out, edge_label) + a * kd_loss
        loss.backward()
        optimizer.step()

        y_val, p_val = collect_scores_lp(model, val_data)
        val_auc, val_ap = evaluate_auc_aupr(y_val, p_val)

        y_test, p_test = collect_scores_lp(model, test_data)
        test_auc, test_ap = evaluate_auc_aupr(y_test, p_test)

        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_test_ap  = test_ap

        # Uncomment for per-epoch logs
        # print(f'epoch {epoch:03d} loss {loss.item():.6f} thr {best_t:.3f} | '

    end = time.perf_counter()
    run_time = end - start
    print("Runtime:", run_time)
    print('KD_Test')
    print('*********************************************************')
    print('final_test_auc: {:.4f} | final_test_ap : {:.4f} | best_val_auc: {:.4f}'.format(
              final_test_auc, final_test_ap, best_val_auc))
    return {
        'final_test_auc': final_test_auc,
        'final_test_ap':  final_test_ap,
        'best_val_auc':   best_val_auc
    }

# ============================================================
#                  Training entry point
# ============================================================
if __name__ == "__main__":
    # ---- 1) Train GNN (optional backbones: "gcn" / "sage" / "gat" / "cheb") ----
    T_model = GNN(dataset.num_features, 128, 64, backbone="gcn").to(device)
    print(T_model)
    GNN_Representation = fit_GNN(T_model, epochs=100)

    # Detach gradients for KD
    GNN_Representation = GNN_Representation.detach()

    # ---- 2) Train the MLP baseline ----
    S_model = MLP(dataset.num_features, 128, 64).to(device)
    print(S_model)
    fit_MLP(S_model, epochs=100)

    # ---- 3) Compute the representation gap between GNN and MLP (optional) ----
    mse_loss = nn.MSELoss()
    MLP_Rep = S_model.MLP_Encoded_Result
    if MLP_Rep is not None:
        mse_value = mse_loss(GNN_Representation, MLP_Rep)
        kl_value  = F.kl_div(MLP_Rep.softmax(dim=-1).log(), GNN_Representation.softmax(dim=-1), reduction='batchmean')
        similarity = torch.cosine_similarity(GNN_Representation, MLP_Rep, dim=0)
        average_similarity = torch.mean(similarity)
        print('mse_value:', mse_value.item())
        print('kl_value:', kl_value.item())
        print('average_similarity:', average_similarity.item())

    # ---- 4) Distillation training (using GNN representations as the teacher) ----
    P_model1 = Projector(64, 64).to(device)
    P_model2 = Projector(64, 64).to(device)
    P_model3 = Projector(64, 64).to(device)
    kd_result = KnowledgeDistillation(
        model=S_model,
        P_model1=P_model1,
        P_model2=P_model2,
        P_model3=P_model3,
        dataset=dataset,
        epochs=100,
        a=1,  # Your original hyperparameter: put all weight on KL; reduce it if needed, e.g. 0.1 to 0.5
        GNN_Representation=GNN_Representation
    )
    print(S_model)