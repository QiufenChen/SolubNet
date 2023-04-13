# SolubNet
A self-developed graph convolutional network (GCN) architecture, SolubNet, for drugs aqueous solubility prediction.

## SolubNet Model
Figure 1 illustrates the structure of SoubNet. It is a three-layer TAGCN. Each layer contains 32 units and a rectified linear unit (ReLU) activation function. The detailed workflow is given in Figure 2(A) and Figure 2(B).

![Figure-1](https://user-images.githubusercontent.com/52032167/231805942-5de0aee4-ca8a-4eac-88c0-6719a86be7b2.png)
<p align="center">Figure 1 The structure of SolubNet</p>

Here, we use layer-wise relevance propagation (LRP) to explain how input features are relevant to the decision of neural network. LRP is a powerful method that is able to unambiguously decompose the output into atomic contributions. Figure 2(C) demonstrates the interpretation process of SolubNet prediction.

![Figure-2](https://user-images.githubusercontent.com/52032167/231806087-e39814df-b401-4cd0-8c05-dc0f7663dd23.png)
<p align="center">Figure 2 The workflow of SolubNet predictive model with interpretability</p>

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

### Evaluate model
1. If you want to use the SolubNetD:
```python
  python SolubNetD/test.py
```
2. If you want to use the SolubNetH:
```python
  python SolubNetH/ test.py
```
3. If you want to use the SolubNetC:
```python
  python SolubNetC/test.py
```

### Contributing to the project
Any pull requests or issues are welcome.

### Progress
README for running SolubNet.
