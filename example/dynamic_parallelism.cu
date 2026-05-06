#include <cuda_runtime.h>
#include <stdio.h>

// Device function for quicksort partition
__device__ int partition(int* data, int low, int high) {
  int pivot = data[high];
  int i = low - 1;
  for (int j = low; j < high; j++) {
    if (data[j] < pivot) {
      ++i;
      int temp = data[i];
      data[i] = data[j];
      data[j] = temp;
    }
  }

  int temp = data[i + 1];
  data[i + 1] = data[high];
  data[high] = temp;

  return i + 1;
}

// Child kernel for quicksort

__global__ void quickSortKernel(int* data, int low, int high) {
  if (low < high) {
    int pi = partition(data, low, high);

    // Launch quicksort on partitions recursively
    // CDP 2.0 (sm_90+): child kernels implicitly synchronize
    // when the parent thread block completes
    quickSortKernel<<<1, 1>>>(data, low, pi - 1);
    quickSortKernel<<<1, 1>>>(data, pi + 1, high);
  }
}

// Quicksort wrapper
void quickSort(int* data, int size) {
  quickSortKernel<<<1, 1>>>(data, 0, size - 1);
  cudaDeviceSynchronize();
}

// Kernel for BFS traversal
__global__ void bfsKernel(int* adjacency, bool* visited, int* queue, int* front,
                          int* rear, int nodeCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nodeCount && visited[idx]) {
    for (int i = 0; i < nodeCount; ++i) {
      if (adjacency[idx * nodeCount + i] && !visited[i]) {
        visited[i] = true;
        int pos = atomicAdd(rear, 1);
        queue[pos] = i;

        // Launch child kernel recursively
        // CDP 2.0: implicit sync at thread block exit
        bfsKernel<<<1, 1>>>(adjacency, visited, queue, front, rear, nodeCount);
      }
    }
  }
}

// BFS wrapper
void bfs(int* adjacency, bool* visited, int* queue, int nodeCount) {
  int* d_front;
  int* d_rear;
  cudaMalloc(&d_front, sizeof(int));
  cudaMalloc(&d_rear, sizeof(int));

  int zero = 0, one = 1;
  bool trueVal = true;
  cudaMemcpy(d_front, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rear, &one, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&queue[0], &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&visited[0], &trueVal, sizeof(bool), cudaMemcpyHostToDevice);

  bfsKernel<<<1, nodeCount>>>(adjacency, visited, queue, d_front, d_rear,
                              nodeCount);
  cudaDeviceSynchronize();

  cudaFree(d_front);
  cudaFree(d_rear);
}

// Main entry point

int main() {
  int h_data[] = {10, 7, 8, 9, 1, 1};
  int size = sizeof(h_data) / sizeof(h_data[0]);

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));
  cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

  // Quicksort execution
  quickSort(d_data, size);
  cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Output sorted array
  printf("Sorted array: ");
  for (int i = 0; i < size; ++i) {
    printf("%d ", h_data[i]);
  }
  printf("\n");

  // Graph adjacency matrix for BFS
  const int nodeCount = 4;
  int h_adjacency[nodeCount * nodeCount] = {0, 1, 0, 1, 1, 0, 1, 0,
                                            0, 1, 0, 1, 1, 0, 1, 0};

  bool h_visited[nodeCount] = {false};
  //   int h_queue[nodeCount];

  int* d_adjacency;
  bool* d_visited;
  int* d_queue;
  cudaMalloc(&d_adjacency, nodeCount * nodeCount * sizeof(int));
  cudaMalloc(&d_visited, nodeCount * sizeof(bool));
  cudaMalloc(&d_queue, nodeCount * sizeof(int));

  cudaMemcpy(d_adjacency, h_adjacency, nodeCount * nodeCount * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_visited, h_visited, nodeCount * sizeof(bool),
             cudaMemcpyHostToDevice);

  // BFS execution
  bfs(d_adjacency, d_visited, d_queue, nodeCount);

  cudaMemcpy(h_visited, d_visited, nodeCount * sizeof(bool),
             cudaMemcpyDeviceToHost);

  // output visited nodes
  printf("Visited nodes: ");
  for (int i = 0; i < nodeCount; ++i) {
    if (h_visited[i]) {
      printf("%d ", i);
    }
  }
  printf("\n");

  cudaFree(d_data);
  cudaFree(d_adjacency);
  cudaFree(d_visited);
  cudaFree(d_queue);

  return 0;
}