# SolubNet
A self-developed graph convolutional network (GCN) architecture, SolubNet, for drugs aqueous solubility prediction.

## SolubNet Model
Figure 1 illustrates the structure of SoubNet. It is a three-layer TAGCN. Each layer contains 32 units and a rectified linear unit (ReLU) activation function. The detailed workflow is given in Figure 2(A) and Figure 2(B).

![Figure-1](https://user-images.githubusercontent.com/52032167/231805942-5de0aee4-ca8a-4eac-88c0-6719a86be7b2.png)
<p align="center"> Figure 1 The structure of SolubNet</p>

Here, we use layer-wise relevance propagation (LRP) to explain how input features are relevant to the decision of neural network. LRP is a powerful method that is able to unambiguously decompose the output into atomic contributions. Figure 2(C) demonstrates the interpretation process of SolubNet prediction.

![Figure-2](https://user-images.githubusercontent.com/52032167/231806087-e39814df-b401-4cd0-8c05-dc0f7663dd23.png)
<p align="center"> Figure 2 The workflow of SolubNet predictive model with interpretability</p>

## Quick Start
### Requirements
Python==3.8 \
pytorch==1.12.1 \
rdkit==2022.9.5 \
scikit-learn==1.2.2 \
dgl==1.0.1

### Download SolubNet
```bash
  git clone https://github.com/QiufenChen/SolubNet.git
```

### Dataset Preparation
In this project, the three benchmark datasets (Delaney1144, Hou1289, and Cui9943) are employed to train the SolubNet (10-fold cross validation), named as SolubNetD, SolubNetH, and SolubNetC respectively. Three independent test sets (Yalk21, Cui62, and Klop120) are used to verify the generalization ability of the model. If you want to directly use our model to test new data, please change the data path in test.py yourself. Please use the best model we have reported in the paper.

Here, we have provided a well organized benchmark dataset (see **BenchmarkDatasets**), including scaffold-split and random-split, which can be directly used to train any model. Independent test sets are also provided (see **IndependentDatasets**).

### Train model
1. If you want to train the SolubNetD:
```python
  python SolubNetD/TenFold.py > train.log
```
2. If you want to train the SolubNetH:
```python
  python SolubNetH/TenFold.py > train.log
```
3. If you want to train the SolubNetC:
```python
  python SolubNetC/TenFold.py > train.log
```

### Evaluate model
1. If you want to test the SolubNetD:
```python
  python SolubNetD/test.py > test.log
```
2. If you want to test the SolubNetH:
```python
  python SolubNetH/test.py > test.log
```
3. If you want to test the SolubNetC:
```python
  python SolubNetC/test.py > test.log
```

### Interpretability of SolubNetC
```python
  python SolubNetC/drawAtomWeight.py
```

### Contributing to the project
Any pull requests or issues are welcome  to contact chenqf829@foxmail.com.

### Progress
README for running SolubNet.

### Citation
```bash
@article{CHEN2023100010,
title = {An interpretable graph representation learning model for accurate predictions of drugs aqueous solubility},
journal = {Artificial Intelligence Chemistry},
volume = {1},
number = {2},
pages = {100010},
year = {2023},
issn = {2949-7477},
doi = {https://doi.org/10.1016/j.aichem.2023.100010},
url = {https://www.sciencedirect.com/science/article/pii/S2949747723000106},
author = {Qiufen Chen and Yuewei Zhang and Peng Gao and Jun Zhang},
keywords = {Drug aqueous solubility, Topology adaptive graph convolutional networks, Layer-wise relevance propagation},
abstract = {As increasingly more data science-driven approaches have been applied for compound properties predictions in the domain of drug discovery, such kinds of methods have displayed considerable accuracy compared to conventional ones. In this work, we proposed an interpretable graph learning representation model, SolubNet, for drug aqueous solubility prediction. The comprehensive evaluation demonstrated that SolubNet can successfully capture the quantitative structure-property relationship and can be interpreted with layer-wise relevance propagation (LRP) algorithm regarding how prediction values are generated from original input structures. The key advantage of SolubNet lies in the fact that it includes 3 layers of Topology Adaptive Graph Convolutional Networks which can efficiently perceive chemical local environments. SolubNet showed high performance in several tasks for drugs’ aqueous solubility prediction. LRP revealed that SolubNet can identify high and low polar regions of a given molecule, assigning them reasonable weights to predict the final solubility, in a way highly compatible with chemists’ intuition. We are confident that such a flexible yet interpretable and accurate tool will largely enhance the efficiency of drug discovery, and will even contribute to the methodology development of computational pharmaceutics.}
```
}
