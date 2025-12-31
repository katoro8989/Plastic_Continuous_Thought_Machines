# Differences from Original CTM

This repository extends the original Continuous Thought Machine (CTM) with hypernetwork support, enabling dynamic weight adjustment based on input state.

## Motivation: Dynamic Weights for Plasticity

Standard recurrent neural networks, including the original CTM, use fixed weight matrices during inference. This means their processing rules remain static regardless of input changes or thought process evolution, lacking a mechanism to dynamically adapt to context changes.

In contrast, the human brain exhibits **Short-Term Plasticity (STP)**, where synaptic connections undergo continuous dynamic changes not only during long-term learning but also across short timescales (seconds to minutes) during thought and inference. The neocortex dynamically alters transmission efficiency based on recent activity history, functioning like filters that adaptively process information based on context.

This implementation brings similar dynamic weight capabilities to CTM, allowing the model to autonomously switch information-processing modes based on the current state.

## Key Differences

The main enhancements are the implementations of:
- **`HyperSynapseUNET`** (see `models/modules.py:221-222`): Hypernetwork support for Synapse modules
- **`HyperSuperLinear`** (see `models/modules.py:506`): Hypernetwork support for Neuron-Level Models (NLM)

Both allow you to optionally use hypernetworks via command-line arguments.

### What This Enables

#### Synapse Hypernetworks
- **Optional hypernetwork usage**: Control whether to use hypernetworks through the `--use_hyper` flag and `--hyper_layers` argument
- **Flexible layer selection**: Choose which layers to apply hypernetworks to:
  - `'none'`: No hypernetwork (equivalent to original CTM)
  - `'bottleneck'`: Only bottleneck layers (recommended)
  - `'down'`: All down projection layers
  - `'up'`: All up projection layers
  - `'all'`: All layers

#### NLM Hypernetworks
- **Neuron-Level Model hypernetworks**: Apply hypernetworks to NLM modules via `--use_hyper_nlm` flag
- **Adaptive temporal dynamics**: Each neuron's update rule can be dynamically adjusted based on current state
- **Shared hypernetwork**: All neurons share the same hypernetwork for parameter efficiency

#### Backward Compatibility
Setting `hyper_layers='none'` and omitting `--use_hyper` and `--use_hyper_nlm` results in the same behavior as the original CTM.

### Mathematical Formulation

The dynamic weight mechanism works by adding a context-dependent component to the base weights:

```
W_t = W_base + ΔW(z_t)
```

where:
- `W_base` is the fixed base weight matrix
- `ΔW(z_t)` is the dynamic component that depends on the input `z_t` at time `t`

To reduce computational cost, we use a **low-rank approximation**:

```
ΔW(z_t) = U(φ(z_t)) · V(φ(z_t))^T
```

where:
- `φ` is a context compressor (single-layer MLP with Layer Normalization) that transforms `z_t` into a lower-dimensional representation
- `U` and `V` are linear layers that generate the low-rank matrices from the compressed context
- The rank is controlled by the `hyper_rank` parameter (typically 4-16)

This low-rank decomposition significantly reduces memory requirements while maintaining the ability to adapt weights dynamically.

### Implementation Details

The `HyperContinuousThoughtMachine` class inherits from `ContinuousThoughtMachine` and adds:
- **Synapse hypernetwork parameters**:
  - `hyper_layers`: Controls which layers use hypernetworks
  - `hyper_rank`: LoRA rank for the hypernetwork decomposition
- **NLM hypernetwork parameters**:
  - `use_hyper_nlm`: Boolean flag to enable NLM hypernetworks
  - `hyper_nlm_rank`: LoRA rank for NLM hypernetwork decomposition

**Architecture Components:**
- **Context Compressor (`φ`)**: A single-layer MLP with Layer Normalization that compresses the input state into a lower-dimensional representation
- **Hypernetwork Heads (`U`, `V`)**: Linear layers that generate the low-rank weight matrices from the compressed context
- **Gate Mechanism**: Optional learnable gating that controls the strength of the dynamic component

**Layer Selection Strategy:**
To balance expressiveness and parameter efficiency, hypernetworks can be applied selectively:
- **Bottleneck layers** (recommended): Applied only to the central layers where information is most compressed, providing maximum impact with minimal parameter overhead
- **All layers**: Maximum expressiveness but higher computational cost

This design allows seamless switching between standard CTM and hypernetwork-enhanced CTM (for both Synapse and NLM) without code changes, simply by adjusting command-line arguments.

