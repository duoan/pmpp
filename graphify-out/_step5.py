import sys, json
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

labels = {
    0: "NN Layer Implementations",
    1: "Heat Equation Simulation",
    2: "Conv2D Backward & Utils",
    3: "Autograd Framework",
    4: "Model & Sequential Training",
    5: "Quadtree Visualization",
    6: "Loss Functions",
    7: "Bezier Curve Tessellation",
    8: "PMPP Chapter Overview",
    9: "Parallel Primitives & Sparse",
    10: "Sparse Format Diagrams",
    11: "CUDA Library Wrappers",
    12: "Exercise Diagrams",
    13: "BFS & Graph Infrastructure",
    14: "Benchmark Entry Points",
    15: "Vector Addition CUDA",
    16: "Gaussian Blur CUDA",
    17: "MRI Reconstruction (FHD)",
    18: "Prefix Scan Exercises",
    19: "FFT & Convolution Visuals",
    20: "2D Convolution Kernel",
    21: "Graph BFS Algorithms",
    22: "Dynamic Parallelism ctypes",
    23: "Memory Locality & Tiling",
    24: "GPU Sorting Algorithms",
    25: "Parallel Merge Exercise",
    26: "Conjugate Gradient Solver",
    27: "Matrix-Vector Multiplication",
    28: "Matrix Multiply Row vs Col",
    29: "Matrix Multiplication CUDA",
    30: "RGB to Grayscale",
    31: "BFS Traversal (Python)",
    32: "cuDNN Wrapper",
    33: "Vector Add Benchmarks",
    34: "MNIST Predictions",
    35: "Stencil Kernel",
    36: "Conv Ops C Extension",
    37: "Coulomb Energy Benchmark",
    38: "Architecture & Reduction",
    39: "Convolution & Deep Learning",
    40: "CUDA Dynamic Parallelism",
    41: "Graph BFS Chapter",
    42: "ELL Sparse Format",
    43: "Tiled GEMM Exercise",
    44: "Benchmark Module",
    45: "Coulomb Energy Module",
    46: "Radix Sort Module",
    47: "Merge Sort Module",
    48: "Utils Module",
    49: "Init Module",
    50: "Setup Module",
    51: "Julia Vector Add",
    52: "Conv2D Loader",
    53: "Filter Radius Parser",
    54: "Stencil API (C)",
    55: "Stencil Tile Rationale",
    56: "Heat CUDA Wrapper",
    57: "Heat Simulation Manager",
    58: "Heat Animator",
    59: "Radix Sort Header",
    60: "Merge Sort Header",
    61: "BFS Queue Traversal",
    62: "Device Memory Alloc",
    63: "Graph Generators",
    64: "BFS Compare Utility",
    65: "CG Solver Implementation",
    66: "Flatten Layer",
    67: "Sigmoid Layer",
    68: "ReLU Layer",
    69: "Pooling Setup",
    70: "Multidimensional Grids",
}

questions = suggest_questions(G, communities, labels)

report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, '.', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report)
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}))
print('Report updated with community labels')
