#include <stdlib.h>

#include "../include/bfs_sequential.h"
#include "../include/graph_structures.h"

// BFS returning a pointer to the list of levels for all vertices
int* bfs(const CSRGraph& graph, int startingNode) {
    int* levels = (int*)malloc(sizeof(int) * graph.numVertices);
    unsigned char* visited = (unsigned char*)calloc(graph.numVertices, sizeof(unsigned char));
    int* queue = (int*)malloc(sizeof(int) * graph.numVertices);
    int queueHead = 0;
    int queueTail = 0;

    // set the default level to -1 meaning it is not yet visited
    for (int i = 0; i < graph.numVertices; i++) {
        levels[i] = -1;
    }

    levels[startingNode] = 0;
    visited[startingNode] = 1;
    queue[queueTail++] = startingNode;

    while (queueHead < queueTail) {
        int vertex = queue[queueHead++];

        for (int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
            int neighbour = graph.dst[edge];
            if (!visited[neighbour]) {
                levels[neighbour] = levels[vertex] + 1;
                visited[neighbour] = 1;
                queue[queueTail++] = neighbour;
            }
        }
    }
    free(visited);
    free(queue);
    return levels;
}
