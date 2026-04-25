# MASC 515 Assignment 3: AI and microgpt

This repository contains an enhanced implementation of **microgpt**, a dependency-free, pure Python GPT model. This project was completed as part of the MASC 515 course at the University of Southern California (USC).

The original `microgpt.py` has been extended to include modern transformer architectural improvements: **GELUs**, **LoRA**, **RoPE**, and **Mixture of Experts (MoE)**. 

**The project is helped by Gemini 3.1 Pro.**

## Implemented Algorithms

### 1. Gaussian Error Linear Units (GELUs)
* **Underlying Idea**: Unlike the standard ReLU which maps all negative inputs to zero, GELU weights the input by its percentile under a Gaussian distribution. This provides a smoother, non-monotonic activation that helps models learn complex patterns more effectively.
* **Mathematical Form**: $GELU(x) = x \Phi(x) = x \cdot P(X \le x)$, where $\Phi(x)$ is the standard normal cumulative distribution function.
* **Implementation**: Added a `.gelu()` method to the `Value` class in `microgpt.py`. It uses `math.erf` to calculate the CDF and implements the corresponding derivative for the autograd backward pass.

### 2. Low-Rank Adaptation (LoRA)
* **Underlying Idea**: LoRA is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture. This significantly reduces the number of trainable parameters while maintaining performance.
* **Implementation**: For the Attention weights ($W_q, W_k, W_v$), we introduced two low-rank matrices $A$ and $B$. The forward pass is modified to $h = Wx + BAx$. Only $A$ and $B$ are updated to adapt the model.

### 3. Rotary Position Embedding (RoPE)
* **Underlying Idea**: RoPE encodes absolute positional information with a rotation matrix and naturally incorporates relative positional dependency in self-attention formulation. Unlike absolute position embeddings, RoPE allows the model to generalize to longer sequences by rotating the query and key vectors.
* **Implementation**: Removed the learned absolute position embedding (`wpe`). Implemented the `apply_rope` function that rotates pairs of dimensions in the query and key vectors based on their token index and a frequency scale.

### 4. Mixture of Experts (MoE)
* **Underlying Idea**: MoE increases model capacity (number of parameters) without a proportional increase in computational cost per token. It replaces the dense MLP layer with a set of sparse "experts" and a "router" that selects which experts process each token.
* **Implementation**: Replaced the standard MLP block with a structure featuring 4 experts. A "Router" linear layer calculates gating scores for each token, and a **Top-2** selection mechanism routes the token through the most relevant experts.

## Setup and Usage

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    ```
2.  **Run Training**:
    The code uses `names.txt` to train a character-level GPT.
    ```bash
    python microgpt.py
    ```

## References
* [1] Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv:1606.08415*.
* [2] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
* [3] Su, J., et al. (2021). RoFormer: Sinusoidal Embeddings with Trainable Rotation. *arXiv:2104.09864*.
* [4] Hugging Face. (2023). Mixture of Experts Explained.
