# Graph Report - .  (2026-04-10)

## Corpus Check
- 58 files · ~798,361 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 238 nodes · 258 edges · 38 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Layer` - 11 edges
2. `Sequential` - 9 edges
3. `Linear` - 9 edges
4. `Sigmoid` - 9 edges
5. `ReLU` - 9 edges
6. `MaxPooling2D` - 8 edges
7. `Flatten` - 8 edges
8. `Conv2D` - 8 edges
9. `QuadtreeVisualizer` - 7 edges
10. `HeatEquationCUDA` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Train a neural network model on the provided data.      Args:         model: The` --uses--> `Sequential`  [INFERRED]
  chapter-16/code/autograd_manual/examples/xor_example.py → chapter-16/code/autograd_manual/nn/model.py
- `Train a neural network model on the provided data.      Args:         model: The` --uses--> `MSELoss`  [INFERRED]
  chapter-16/code/autograd_manual/examples/xor_example.py → chapter-16/code/autograd_manual/nn/loss.py
- `Conv2D` --uses--> `Layer`  [INFERRED]
  chapter-16/code/autograd_manual/nn/layers/conv.py → chapter-16/code/autograd_manual/nn/layers/base.py
- `Train a neural network model on the provided data.      Args:         model: The` --uses--> `Linear`  [INFERRED]
  chapter-16/code/autograd_manual/examples/xor_example.py → chapter-16/code/autograd_manual/nn/layers/linear.py
- `Train a neural network model on the provided data.      Args:         model: The` --uses--> `Sigmoid`  [INFERRED]
  chapter-16/code/autograd_manual/examples/xor_example.py → chapter-16/code/autograd_manual/nn/layers/activations.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (7): Flatten, ReLU, Sigmoid, Layer, Layer, Linear, MaxPooling2D

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (19): HeatAnimator, HeatEquationCUDA, HeatSimulation, main(), 3D Heat Equation Simulation Manager, Initialize simulation parameters          Args:             N: Grid size (N×N×N), Create initial temperature distribution, Advance simulation by one time step (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (11): backward(), Linear, LinearFunction, main(), MSELoss, SGD, Sigmoid, SigmoidFunction (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.2
Nodes (8): main(), QuadtreeVisualizer, Initialize the quadtree visualizer with CUDA library, Visualize the original points and quadtree division, Recursively draw quadtree boundaries (approximation), Main demonstration function, Generate sample 2D points for testing, Build quadtree using CUDA implementation

### Community 4 - "Community 4"
Cohesion: 0.2
Nodes (11): analyze_tessellation_comparison(), benchmark_performance(), BezierLineC, plot_curves_comparison(), Compare static vs dynamic parallelism results, Dynamic parallelism tessellation using CUDA dynamic parallelism      Args:, Analyze and compare static vs dynamic results, Simple performance benchmark using triton (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.2
Nodes (2): CrossEntropyLoss, MSELoss

### Community 6 - "Community 6"
Cohesion: 0.2
Nodes (5): Adam, SGD, Train a neural network model on the provided data.      Args:         model: The, train_model(), xor_example()

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (2): cleanup_cublas(), Clean up cuBLAS resources

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (1): Sequential

### Community 9 - "Community 9"
Cohesion: 0.29
Nodes (1): Conv2D

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (0): 

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (0): 

### Community 12 - "Community 12"
Cohesion: 0.6
Nodes (3): compile_extension(), main(), vector_multipication_loop()

### Community 13 - "Community 13"
Cohesion: 0.5
Nodes (2): compile_extension(), main()

### Community 14 - "Community 14"
Cohesion: 0.83
Nodes (3): get_filter_radius_from_header(), load_conv2d_extension(), main()

### Community 15 - "Community 15"
Cohesion: 0.67
Nodes (2): conjugate_gradient(), Solve the linear system Ax = b using the Conjugate Gradient method.      Paramet

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (2): compile_extension(), main()

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): compile_extension(), main()

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (2): compile_extension(), main()

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (2): compile_extension(), main()

### Community 20 - "Community 20"
Cohesion: 0.67
Nodes (2): dfs(), we take a matrix in CRS format as an input and we return levels for all elements

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (2): cleanup_cudnn(), Clean up cuDNN resources

### Community 22 - "Community 22"
Cohesion: 0.67
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **31 isolated node(s):** `Solve the linear system Ax = b using the Conjugate Gradient method.      Paramet`, `BezierLineC`, `Dynamic parallelism tessellation using CUDA dynamic parallelism      Args:`, `Static parallelism tessellation (original version)      Args:         control_po`, `Compare static vs dynamic parallelism results` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 23`** (2 nodes): `stencil.h`, `cdiv()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `mnist_example.py`, `mnist_cnn_example()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `conv_ops.c`, `convLayer_backward_x_grad()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `benchmark.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `cenergy.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `gpu_radix_sort.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `gpu_merge_sort.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `utils.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): ` __init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `_step5.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `_step4.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `_step9.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `_step6.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `_merge.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Train a neural network model on the provided data.      Args:         model: The` connect `Community 6` to `Community 0`, `Community 8`, `Community 5`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `Layer` connect `Community 0` to `Community 9`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `MSELoss` connect `Community 5` to `Community 6`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Layer` (e.g. with `MaxPooling2D` and `Linear`) actually correct?**
  _`Layer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Linear` (e.g. with `Layer` and `Train a neural network model on the provided data.      Args:         model: The`) actually correct?**
  _`Linear` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Sigmoid` (e.g. with `Layer` and `Train a neural network model on the provided data.      Args:         model: The`) actually correct?**
  _`Sigmoid` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ReLU` (e.g. with `Layer` and `Train a neural network model on the provided data.      Args:         model: The`) actually correct?**
  _`ReLU` has 2 INFERRED edges - model-reasoned connections that need verification._