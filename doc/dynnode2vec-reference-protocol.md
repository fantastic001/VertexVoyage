# DynNode2Vec evaluation methodology

Reference evaluation protocol from the original DynNode2Vec paper, used as the
template for VertexVoyage's downstream tasks (see
[evaluation.md](evaluation.md)).

DynNode2Vec uses Logistic Regression with Hadamard product for link prediction and Linear SVM for node classification, evaluated on three dynamic, real-world datasets. The evaluation datasets consist of the Hep-th co-authorship network, the Autonomous Systems (AS) routing network, and the Enron email communication network. For details on the evaluation methodology, see the study available at Springer.

https://link.springer.com/article/10.1007/s41060-022-00376-3

Logistic regression here is defined as follows:

$$ P(y=1|u,v) = \frac{1}{1 + e^{-w^T (f(u) \odot f(v))}} $$

Where $f(u)$ and $f(v)$ are the embeddings of nodes $u$ and $v$, $\odot$ denotes the Hadamard product, and $w$ is the weight vector learned during training. The model predicts the probability of a link existing between nodes $u$ and $v$ based on their embeddings.

Hadamard product is an element-wise multiplication of two vectors, defined as:

$$ (f(u) \odot f(v))_i = f(u)_i \cdot f(v)_i $$

For node classification, Linear SVM is used to classify nodes based on their embeddings. The SVM finds a hyperplane that best separates the classes in the embedding space, maximizing the margin between different classes. The performance of both link prediction and node classification is evaluated using standard metrics such as AUC for link prediction and accuracy for node classification.

