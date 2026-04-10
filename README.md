# KDBoost4LinkPrediction
## Abstract
Link prediction is crucial for understanding complex network structures and evolutionary mechanisms. In recent years, feature-based knowledge distillation (KD) has shown strong potential for improving link prediction performance. However, its reliance on a fixed GNN teacher model limits its adaptability across different networks. Moreover, direct feature matching may cause the student model to overfit. To address these issues, we propose KDBoost, a framework that enhances feature-based \underline{KD} to \underline{Boost} the performance of link prediction. KDBoost leverages Graph Neural Architecture Search (GNAS) to automatically select the optimal teacher model for each network. In addition, it introduces a Projector module to transforms student features, thereby alleviating overfitting and improving generalization. Experiments on eight benchmark networks demonstrate that KDBoost achieves average improvements of 7.6\%, 7.4\%, and 10.5\% in AUC, AUPR, and F1-score, respectively, over traditional feature distillation methods. Compared with recent GNN distillation methods, KDBoost further improves these metrics by 3.4\%, 3.1\%, and 4.9\%, indicating its effectiveness for accurate and generalizable link prediction. Our code is available at https://github.com/yabingyao/KDBoost4LinkPrediction.git.

##  Method Overview
![GitHub图像](/img/fig1.jpg)

Overview of the proposed KDBoost framework.  
(a) An example network with 6 nodes, including its topology (upper left) and node feature matrix (lower left).  
(b) The upper-middle part illustrates the workflow of GNAS, which selects the optimal GNN architecture for the given network from the predefined search space and trains it as the teacher model. The resulting node representation is denoted by $`\mathbf{t}_u`$.  
(c) The lower-middle part shows the two-layer MLP used as the student model, which outputs the initial node representation $`\mathbf{s}_u`$.  
(d) The right part depicts the Projector, which performs a nonlinear transformation on the student representation to obtain $`\tilde{\mathbf{s}}^{\,l}_u`$ for feature distillation.
 
##  Experimental setup

numpy==1.23.5

scikit_learn==1.2.1

scikit_learn==1.2.0

torch==1.13.1

torch_cluster==1.6.0

torch_geometric==2.2.0

torch_sparse==0.6.16+pt113cu117

## Run Setting  
All experiments use a fixed random seed and validation split; the teacher GNN is searched from a two-layer differentiable architecture search space with 9 candidate aggregation operations, all model output dimensions are set to 128, and the configurations of Graph2Feat and GLNN follow their original settings except for teacher selection. 

```python GraphNAS4Teacher.py```

```python KDBooxt.py```

