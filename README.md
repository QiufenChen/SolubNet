# SolubNet
A self-developed graph convolutional network (GCN) architecture, SolubNet, for drugs aqueous solubility prediction.

## SolubNet Model
Figure 1 illustrates the structure of SoubNet. It is a three-layer TAGCN. Each layer contains 32 units and a rectified linear unit (ReLU) activation function. The detailed workflow is given in Figure 2(A) and Figure 2(B).
![Figure-1](https://user-images.githubusercontent.com/52032167/201565688-1d80a1a6-dc4d-480a-8028-5f18f24dd742.png)
<p align="center">Figure 1 The structure of SolubNet</p>

Here, we use LRP to explain how input features are relevant to the decision of neural network.  Figure 2(C) demonstrates the interpretation process of SolubNet prediction.
![Figure-2](https://user-images.githubusercontent.com/52032167/201566019-c5dd300a-3207-4681-9c77-982c52c3b784.png)

<p align="center">Figure 2 The workflow of SolubNet predictive model with interpretability</p>

