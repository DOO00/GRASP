# GRASP

GRASP (Global-to-Local Cross-Modal Alignment and Pseudo-Label Purification) is a method for multimodal attributed graph clustering. It aligns cross-modal spectral topology, calibrates reliable local relations across perturbed views, and purifies pseudo-labels with learnable prototypes and a student-teacher model.

## Environment

- Python 3.10 or newer
- PyTorch, NumPy, pandas, SciPy, and scikit-learn
- Install dependencies with `pip install -r requirements.txt`.
- DGL is needed when loading graph files without a cached adjacency matrix. `torch-cluster` is optional for accelerated random walks.
