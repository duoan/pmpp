#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float* x;
    float* y;
} Points;

typedef struct {
    float2 p_min;
    float2 p_max;
} Bounding_box;

typedef struct {
    int id;
    Bounding_box bounding_box;
    int begin;
    int end;
} Quadtree_node;

__host__ __device__ __forceinline__ void points_init(Points* points, float* x, float* y) {
    points->x = x;
    points->y = y;
}

__host__ __device__ __forceinline__ float2 points_get_point(const Points* points, int idx) {
    return make_float2(points->x[idx], points->y[idx]);
}

__host__ __device__ __forceinline__ void points_set_point(Points* points, int idx, float2 p) {
    points->x[idx] = p.x;
    points->y[idx] = p.y;
}

__host__ __device__ __forceinline__ void bounding_box_init(Bounding_box* bbox) {
    bbox->p_min = make_float2(0.0f, 0.0f);
    bbox->p_max = make_float2(1.0f, 1.0f);
}

__host__ __device__ __forceinline__ void bounding_box_compute_center(const Bounding_box* bbox, float2* center) {
    center->x = 0.5f * (bbox->p_min.x + bbox->p_max.x);
    center->y = 0.5f * (bbox->p_min.y + bbox->p_max.y);
}

__host__ __device__ __forceinline__ float2 bounding_box_get_max(const Bounding_box* bbox) {
    return bbox->p_max;
}

__host__ __device__ __forceinline__ float2 bounding_box_get_min(const Bounding_box* bbox) {
    return bbox->p_min;
}

__host__ __device__ __forceinline__ void bounding_box_set(Bounding_box* bbox, float min_x, float min_y, float max_x,
                                                          float max_y) {
    bbox->p_min.x = min_x;
    bbox->p_min.y = min_y;
    bbox->p_max.x = max_x;
    bbox->p_max.y = max_y;
}

__host__ __device__ __forceinline__ void quadtree_node_init(Quadtree_node* node) {
    node->id = 0;
    node->begin = 0;
    node->end = 0;
    bounding_box_init(&node->bounding_box);
}

__host__ __device__ __forceinline__ void quadtree_node_set_id(Quadtree_node* node, int new_id) {
    node->id = new_id;
}

__host__ __device__ __forceinline__ Bounding_box quadtree_node_get_bounding_box(const Quadtree_node* node) {
    return node->bounding_box;
}

__host__ __device__ __forceinline__ void quadtree_node_set_bounding_box(Quadtree_node* node, float min_x, float min_y,
                                                                        float max_x, float max_y) {
    bounding_box_set(&node->bounding_box, min_x, min_y, max_x, max_y);
}

__host__ __device__ __forceinline__ int quadtree_node_num_points(const Quadtree_node* node) {
    return node->end - node->begin;
}

__host__ __device__ __forceinline__ int quadtree_node_points_begin(const Quadtree_node* node) {
    return node->begin;
}

__host__ __device__ __forceinline__ int quadtree_node_points_end(const Quadtree_node* node) {
    return node->end;
}

__host__ __device__ __forceinline__ void quadtree_node_set_range(Quadtree_node* node, int begin, int end) {
    node->begin = begin;
    node->end = end;
}

// Algorithm parameters
struct Parameters {
    // Choose the right set of points to use as in/out
    int point_selector;
    // The number of nodes at a given level (2^k for level k)
    int num_nodes_at_this_level;
    // The recursion depth
    int depth;
    // The max value for depth
    int max_depth;
    // The minimum number of points in a node to stop recursion
    int min_points_per_node;
};

__host__ __device__ __forceinline__ Parameters parameters_create(int max_depth, int min_points_per_node) {
    Parameters params;
    params.point_selector = 0;
    params.num_nodes_at_this_level = 1;
    params.depth = 0;
    params.max_depth = max_depth;
    params.min_points_per_node = min_points_per_node;
    return params;
}

__host__ __device__ __forceinline__ Parameters parameters_next(Parameters params) {
    Parameters next_params;
    next_params.point_selector = (params.point_selector + 1) % 2;
    next_params.num_nodes_at_this_level = 4 * params.num_nodes_at_this_level;
    next_params.depth = params.depth + 1;
    next_params.max_depth = params.max_depth;
    next_params.min_points_per_node = params.min_points_per_node;
    return next_params;
}

// Check the number of points and its depth
__device__ bool check_num_points_and_depth(Quadtree_node* node, Points* points, int num_points, Parameters params) {
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node) {
        // Stop the recursion here. Make sure points[0] contains all the points
        if (params.point_selector == 1) {
            int it = quadtree_node_points_begin(node);
            int end = quadtree_node_points_end(node);
            for (it += threadIdx.x; it < end; it += blockDim.x) {
                if (it < end) {
                    points_set_point(&points[0], it, points_get_point(&points[1], it));
                }
            }
        }
        return true;
    }
    return false;
}

// Count the number of points in each quadrant
__device__ void count_points_in_children(const Points* in_points, int* smem, int range_begin, int range_end,
                                         float2 center) {
    // Initialize shared memory
    if (threadIdx.x < 4) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();
    // Compute the number of points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = points_get_point(in_points, iter);  // Load the coordinates of the point
        if (p.x < center.x && p.y >= center.y) {
            atomicAdd(&smem[0], 1);  // Top-left point?
        }
        if (p.x >= center.x && p.y >= center.y) {
            atomicAdd(&smem[1], 1);  // Top-right point?
        }
        if (p.x < center.x && p.y < center.y) {
            atomicAdd(&smem[2], 1);  // Bottom-left point?
        }
        if (p.x >= center.x && p.y < center.y) {
            atomicAdd(&smem[3], 1);  // Bottom-right point?
        }
    }
    __syncthreads();
}

// Scan quadrants' results to obtain reordering offset
__device__ void scan_for_offsets(int node_points_begin, int* smem) {
    int* smem2 = &smem[4];
    if (threadIdx.x == 0) {
        // smem2 will contain starting positions for writing each quadrant
        smem2[0] = node_points_begin;   // Top-left starts at begin
        smem2[1] = smem2[0] + smem[0];  // Top-right starts after top-left
        smem2[2] = smem2[1] + smem[1];  // Bottom-left starts after top-right
        smem2[3] = smem2[2] + smem[2];  // Bottom-right starts after bottom-left
    }
    __syncthreads();
}

// Reorder points in order to group the points in each quadrant
__device__ void reorder_points(Points* out_points, const Points* in_points, int* smem, int range_begin, int range_end,
                               float2 center) {
    int* smem2 = &smem[4];
    // Reorder points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = points_get_point(in_points, iter);  // Load the coordinates of the point
        int dest = -1;

        // Determine which quadrant the point belongs to
        if (p.x < center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[0], 1);  // Top-left point
        } else if (p.x >= center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[1], 1);  // Top-right point
        } else if (p.x < center.x && p.y < center.y) {
            dest = atomicAdd(&smem2[2], 1);  // Bottom-left point
        } else if (p.x >= center.x && p.y < center.y) {
            dest = atomicAdd(&smem2[3], 1);  // Bottom-right point
        }

        // Move point to its destination
        if (dest >= 0) {
            points_set_point(out_points, dest, p);
        }
    }
    __syncthreads();
}

// Prepare children launch
__device__ void prepare_children(Quadtree_node* children, Quadtree_node* node, Bounding_box bbox, int* smem) {
    if (threadIdx.x == 0) {
        // Points to the bounding-box
        float2 p_min = bounding_box_get_min(&bbox);
        float2 p_max = bounding_box_get_max(&bbox);

        // Compute center for children bounding boxes
        float2 center;
        bounding_box_compute_center(&bbox, &center);

        int* smem2 = &smem[4];  // Starting positions for each quadrant

        // Set up the 4 children only if they have points
        for (int i = 0; i < 4; i++) {
            if (smem[i] > 0) {  // Only set up children that have points
                quadtree_node_set_id(&children[i], i);
                quadtree_node_set_range(&children[i], smem2[i], smem2[i] + smem[i]);

                // Set bounding boxes based on quadrant
                if (i == 0) {  // Top-left
                    quadtree_node_set_bounding_box(&children[i], p_min.x, center.y, center.x, p_max.y);
                } else if (i == 1) {  // Top-right
                    quadtree_node_set_bounding_box(&children[i], center.x, center.y, p_max.x, p_max.y);
                } else if (i == 2) {  // Bottom-left
                    quadtree_node_set_bounding_box(&children[i], p_min.x, p_min.y, center.x, center.y);
                } else {  // Bottom-right
                    quadtree_node_set_bounding_box(&children[i], center.x, p_min.y, p_max.x, center.y);
                }
            }
        }
    }
    __syncthreads();
}

__global__ void build_quadtree_kernel(Quadtree_node* nodes, Points* points, Parameters params) {
    __shared__ int smem[8];  // To store the number of points in each quadrant

    // The current node in the quadtree
    Quadtree_node* node = &nodes[blockIdx.x];

    int num_points = quadtree_node_num_points(node);  // The number of points in the node

    // Check the number of points and its depth
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) {
        return;
    }

    // Compute the center of the bounding box of the points
    Bounding_box bbox = quadtree_node_get_bounding_box(node);
    float2 center;
    bounding_box_compute_center(&bbox, &center);

    // Range of points
    int range_begin = quadtree_node_points_begin(node);
    int range_end = quadtree_node_points_end(node);
    Points* in_points = &points[params.point_selector];             // Input points
    Points* out_points = &points[(params.point_selector + 1) % 2];  // Output points

    // Count the number of points in each child
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan the quadrants' results to know the reordering offset
    scan_for_offsets(quadtree_node_points_begin(node), smem);

    // Move points
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Launch new blocks for children
    if (threadIdx.x == 0) {
        // Check if any child has enough points to subdivide
        bool should_recurse = false;
        for (int i = 0; i < 4; i++) {
            if (smem[i] > params.min_points_per_node && params.depth + 1 < params.max_depth) {
                should_recurse = true;
                break;
            }
        }

        if (should_recurse) {
            // Allocate space for 4 children
            Quadtree_node* children = &nodes[params.num_nodes_at_this_level + blockIdx.x * 4];

            // Prepare children
            prepare_children(children, node, bbox, smem);

            // Launch kernel for each child that has points
            Parameters next_params = parameters_next(params);
            for (int i = 0; i < 4; i++) {
                if (smem[i] > 0) {
                    build_quadtree_kernel<<<1, blockDim.x, 8 * sizeof(int)>>>(&children[i], points, next_params);
                }
            }
        }
    }
}

// Host wrapper function
extern "C" {
int build_quadtree(float* h_x, float* h_y, int num_points, int max_depth, int min_points_per_node, float** result_x,
                   float** result_y, float* bounds, int* num_result_points) {
    // Allocate device memory for points
    float *d_x[2], *d_y[2];
    cudaMalloc(&d_x[0], num_points * sizeof(float));
    cudaMalloc(&d_y[0], num_points * sizeof(float));
    cudaMalloc(&d_x[1], num_points * sizeof(float));
    cudaMalloc(&d_y[1], num_points * sizeof(float));

    // Copy input points to device
    cudaMemcpy(d_x[0], h_x, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[0], h_y, num_points * sizeof(float), cudaMemcpyHostToDevice);

    // Create Points objects
    Points h_points[2];
    points_init(&h_points[0], d_x[0], d_y[0]);
    points_init(&h_points[1], d_x[1], d_y[1]);

    Points* d_points;
    cudaMalloc(&d_points, 2 * sizeof(Points));
    cudaMemcpy(d_points, h_points, 2 * sizeof(Points), cudaMemcpyHostToDevice);

    // Calculate maximum number of nodes needed (conservative estimate)
    int max_nodes = 1;
    for (int i = 1; i <= max_depth; i++) {
        max_nodes += (int)pow(4, i);
    }
    max_nodes *= 2;  // Extra safety margin

    // Allocate device memory for nodes
    Quadtree_node* d_nodes;
    cudaMalloc(&d_nodes, max_nodes * sizeof(Quadtree_node));
    cudaMemset(d_nodes, 0, max_nodes * sizeof(Quadtree_node));

    // Initialize root node
    Quadtree_node root;
    quadtree_node_init(&root);
    quadtree_node_set_id(&root, 0);
    quadtree_node_set_range(&root, 0, num_points);
    quadtree_node_set_bounding_box(&root, bounds[0], bounds[1], bounds[2], bounds[3]);
    cudaMemcpy(d_nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);

    // Create parameters
    Parameters params = parameters_create(max_depth, min_points_per_node);

    // Launch kernel with single block for root
    build_quadtree_kernel<<<1, 32>>>(d_nodes, d_points, params);

    // Wait for completion and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back (final points are in buffer 0)
    *result_x = (float*)malloc(num_points * sizeof(float));
    *result_y = (float*)malloc(num_points * sizeof(float));

    cudaMemcpy(*result_x, d_x[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*result_y, d_y[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);

    *num_result_points = num_points;

    // Cleanup
    cudaFree(d_x[0]);
    cudaFree(d_y[0]);
    cudaFree(d_x[1]);
    cudaFree(d_y[1]);
    cudaFree(d_points);
    cudaFree(d_nodes);

    return 0;
}
}
