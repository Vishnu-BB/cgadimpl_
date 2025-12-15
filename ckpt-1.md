# Activation Checkpointing Module

## Overview
This module implements **activation checkpointing** (also known as gradient checkpointing) for the `cgad` autograd engine. 
Checkpointing is a memory-optimization technique used during the training of deep neural networks. Instead of storing *all* intermediate activations for the backward pass (which consumes O(N) memory), we store only a subset of "checkpoint" nodes and recompute the missing activations on-the-fly during backpropagation.

This trades **computation** (re-running forward ops) for **memory** (storing fewer tensors).

## Files
- **Header**: `include/ad/checkpoint.hpp`
- **Implementation**: `src/core/checkpoint.cpp`

## Key Features

### 1. Configurable Checkpointing (`CheckpointOptions`)
The system allows fine-grained control via `CheckpointOptions`:
- `save_inputs`: Stores input tensors of the checkpointed node to facilitate recomputation.
- `force`: Forces a node to be a checkpoint even if heuristics suggest otherwise.
- `verbose`: Enables debug logging.
- `save_rng`: **(Planned)** Intended to save the Random Number Generator state to ensure deterministic recomputation of stochastic ops (e.g., Dropout).

### 2. Automatic Strategies
Two utility functions are provided to automatically apply checkpointing to a graph:
- `auto_checkpoint_every_n(root, n)`: Marks every Nth node in the topological traversal.
- `auto_checkpoint_by_depth(root, depth)`: Marks nodes that are deeper than a specific threshold.

### 3. Robust Recomputation Logic
The `recompute_subgraph` function handles the reconstruction of lost data:
- It recursively ensures that all parent nodes of a checkpoint have their values present.
- If a parent is missing its value but is *also* a checkpoint, it triggers a recursive recomputation of that parent.
- It integrates with the `inplace` module via `ag::inplace::on_recomputed` to handle in-place modification tracking.

## Architecture & Implementation Details

### The `Node` Struct
The `Node` struct (in `graph.hpp`) is augmented with:
- `bool is_checkpoint`: Flag indicating if this node is a checkpoint boundary.
- `std::vector<Value> saved_inputs`: Stores strong references to input tensors required to re-run the operation.

### The Recompute Flow
When `backward()` encounters a node with missing data (cleared to save memory):
1. It checks if `is_checkpoint` is true.
2. Calls `checkpoint_impl::recompute_subgraph(node)`.
3. `recompute_subgraph` checks parents:
   - If a parent has data, proceed.
   - If a parent is missing data but is a checkpoint, recurse.
   - **Critical Failure**: If a parent is missing data and is *NOT* a checkpoint, recomputation is impossible. The system logs an error and fails.
4. Once inputs are ready, `forward_eval_node` is called to regenerate the node's `value`.

## Current Limitations & Drawbacks

### 1. RNG State Not Implemented (Stubbed)
**Critical**: The code currently contains stubs for saving and restoring RNG state:
```cpp
// ... (Your RNG saving logic can go here) ...
```
**Impact**: Stochastic operations (like Dropout or random noise layers) will **not** produce the same output during the recomputed forward pass as they did in the original pass. This will lead to incorrect gradients.
**Fix Required**: Implement RNG state serialization in `mark_node_checkpoint` and restoration in `recompute_subgraph`.

### 2. `max_recompute_depth` Ignored
The `CheckpointOptions` struct defines `max_recompute_depth`, but it is **not enforced** in `recompute_subgraph`.
**Risk**: In extremely deep graphs with chained checkpoints, this could lead to a stack overflow during the recursive recomputation calls.

### 3. Strict Dependency Chain
The system requires an unbroken chain of checkpoints. If you have:
`A (Checkpoint) -> B (Normal) -> C (Checkpoint)`
And you clear memory for A, B, and C.
- When C needs recomputation, it needs B.
- B is not a checkpoint, so it cannot recompute itself.
- **Result**: Failure.
**Best Practice**: Ensure that the "gaps" between checkpoints are small enough that their intermediate values are either kept alive or can be trivially reconstructed (though the current logic requires the immediate parent to be present or a checkpoint).

## Memory Management & Potential Leaks

### Strong References in `saved_inputs`
When `mark_node_checkpoint` is called, it copies the input `Value`s into `node->saved_inputs`.
- **Pros**: Guarantees inputs are available for recompute.
- **Cons**: These are strong references. If you checkpoint *too many* nodes, you end up keeping the entire graph's tensors alive, negating the memory benefits.
- **Cycle Risk**: While rare in standard feed-forward/backprop DAGs, if there are reference cycles involving these saved inputs, it could prevent `Node` destruction.

### "Ghost" Memory
If `detach_inputs` is false (default), the saved inputs keep their entire history alive. If you checkpoint a node deep in the graph, it anchors the graph above it. To truly save memory, you typically rely on the fact that non-checkpointed nodes *between* checkpoints will have their values released (if reference counts drop), but the *structure* might remain if anchored by a checkpoint's saved inputs.

## Guide for New Developers

1.  **Fix the RNG**: The highest priority is implementing the RNG state saving/restoring to support Dropout correctly.
2.  **Enforce Depth Limits**: Add a depth counter to `recompute_subgraph` to respect `max_recompute_depth`.
3.  **Optimize Storage**: Consider if `saved_inputs` needs to store the full `Value` (with history) or just the `Tensor` data (if `detach_inputs` is intended to be the default for pure recompute).
4.  **Thread Safety**: The current implementation modifies `node->value` and `node->saved_inputs` without locks. It is **not thread-safe**.

