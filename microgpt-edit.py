"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
Integrated with GELUs, LoRA, RoPE, and MoE.

@karpathy (Original)
Modified for MASC 515 Assignment 3
"""

import os
import math
import random
random.seed(42)

# Dataset preparation
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Autograd Engine
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    
    # 1. Gaussian Error Linear Units (GELUs) Support
    def gelu(self):
        x = self.data
        # Standard Normal CDF approximation using math.erf
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        # Standard Normal PDF for the derivative
        pdf = math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
        # Chain rule: d(x*cdf)/dx = cdf + x*pdf
        return Value(x * cdf, (self,), (cdf + x * pdf,))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Hyperparameters
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
lora_r = 2        # LoRA Rank
n_experts = 4     # Number of MoE Experts

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# Initialize model parameters (Removed wpe for RoPE, Added LoRA and MoE matrices)
state_dict = {'wte': matrix(vocab_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    # 2. LoRA parameters for Attention (W, A, and B matrices)
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wq_a'] = matrix(lora_r, n_embd)
    state_dict[f'layer{i}.attn_wq_b'] = matrix(n_embd, lora_r, std=0.0) # B initialized to 0

    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk_a'] = matrix(lora_r, n_embd)
    state_dict[f'layer{i}.attn_wk_b'] = matrix(n_embd, lora_r, std=0.0)

    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv_a'] = matrix(lora_r, n_embd)
    state_dict[f'layer{i}.attn_wv_b'] = matrix(n_embd, lora_r, std=0.0)

    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    
    # 4. Mixture of Experts parameters
    state_dict[f'layer{i}.moe_router'] = matrix(n_experts, n_embd)
    for e in range(n_experts):
        state_dict[f'layer{i}.expert{e}_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# LoRA-adapted linear transformation
def linear(x, w, wa=None, wb=None):
    out = [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    if wa is not None and wb is not None:
        # LoRA forward pass: Wx + BAx
        x_a = [sum(wai * xi for wai, xi in zip(wao, x)) for wao in wa]
        x_b = [sum(wbi * xai for wbi, xai in zip(wbo, x_a)) for wbo in wb]
        out = [o + b for o, b in zip(out, x_b)]
    return out

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# 3. RoPE (Rotary Position Embedding) Logic
def apply_rope(x, pos_id, n_head, head_dim):
    out = []
    for h in range(n_head):
        for i in range(0, head_dim, 2):
            theta = pos_id / (10000 ** (i / head_dim))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            x0 = x[h * head_dim + i]
            x1 = x[h * head_dim + i + 1]
            out.append(x0 * cos_t - x1 * sin_t)
            out.append(x0 * sin_t + x1 * cos_t)
    return out

def gpt(token_id, pos_id, keys, values):
    x = state_dict['wte'][token_id] # token embedding only (Position emb replaced by RoPE)
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        
        # Apply LoRA via wa and wb arguments
        q = linear(x, state_dict[f'layer{li}.attn_wq'], state_dict[f'layer{li}.attn_wq_a'], state_dict[f'layer{li}.attn_wq_b'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'], state_dict[f'layer{li}.attn_wk_a'], state_dict[f'layer{li}.attn_wk_b'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'], state_dict[f'layer{li}.attn_wv_a'], state_dict[f'layer{li}.attn_wv_b'])
        
        # Apply RoPE to Queries and Keys
        q = apply_rope(q, pos_id, n_head, head_dim)
        k = apply_rope(k, pos_id, n_head, head_dim)
        
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # 2) Mixture of Experts (MoE) block replacing MLP
        x_residual = x
        x = rmsnorm(x)
        
        # Routing mechanism
        router_logits = linear(x, state_dict[f'layer{li}.moe_router'])
        gate_probs = softmax(router_logits)
        
        # Top-2 Expert Routing
        probs_with_idx = [(p, idx) for idx, p in enumerate(gate_probs)]
        probs_with_idx.sort(key=lambda item: item[0].data, reverse=True)
        top_2 = probs_with_idx[:2]
        
        denom = top_2[0][0] + top_2[1][0]
        top_weights = [p / denom for p, idx in top_2]
        top_indices = [idx for p, idx in top_2]
        
        moe_out = [Value(0.0) for _ in range(n_embd)]
        for weight, exp_idx in zip(top_weights, top_indices):
            exp_x = linear(x, state_dict[f'layer{li}.expert{exp_idx}_fc1'])
            # Apply GELU Activation inside experts
            exp_x = [xi.gelu() for xi in exp_x]
            exp_x = linear(exp_x, state_dict[f'layer{li}.expert{exp_idx}_fc2'])
            moe_out = [out + weight * e_x for out, e_x in zip(moe_out, exp_x)]
            
        x = [a + b for a, b in zip(moe_out, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Optimizer
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# Training loop
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")