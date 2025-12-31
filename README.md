# Differences from Original CTM

This repository extends the original Continuous Thought Machine (CTM) with hypernetwork support, enabling dynamic weight adjustment based on input state.

## Key Difference

The main enhancement is the implementation of **`HyperSynapseUNET`** (see `models/modules.py:221-222`), which allows you to optionally use hypernetworks via command-line arguments.

### What This Enables

- **Optional hypernetwork usage**: You can control whether to use hypernetworks through the `--use_hyper` flag and `--hyper_layers` argument
- **Flexible layer selection**: Choose which layers to apply hypernetworks to:
  - `'none'`: No hypernetwork (equivalent to original CTM)
  - `'bottleneck'`: Only bottleneck layers (recommended)
  - `'down'`: All down projection layers
  - `'up'`: All up projection layers
  - `'all'`: All layers
- **Backward compatibility**: Setting `hyper_layers='none'` or omitting `--use_hyper` results in the same behavior as the original CTM

### Implementation Details

The `HyperContinuousThoughtMachine` class inherits from `ContinuousThoughtMachine` and adds:
- `hyper_layers` parameter: Controls which layers use hypernetworks
- `hyper_rank` parameter: LoRA rank for the hypernetwork decomposition

This design allows seamless switching between standard CTM and hypernetwork-enhanced CTM without code changes, simply by adjusting command-line arguments.

For detailed usage instructions, see:
- `HYPER_USAGE.md` - Hypernetwork usage guide
- `HYPER_NLM_USAGE.md` - NLM hypernetwork guide
- `TRAIN_HYPER_EXAMPLES.md` - Training examples

