#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cuda_runtime_api.h>

#include <cstdlib>

__global__ void cudaMoveKernel(
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ destX,
    float* __restrict__ destY,
    float* __restrict__ destR,
    int* __restrict__ wpIndex,
    const int* __restrict__ wpCount,
    const int* __restrict__ wpOffset,
    const float* __restrict__ wpPoolX,
    const float* __restrict__ wpPoolY,
    const float* __restrict__ wpPoolR,
    int numAgents);

void testBlockSizes(
    float*, float*,
    float*, float*, float*,
    int*, const int*, const int*,
    const float*, const float*, const float*,
    int);

        
void testBlockSizes(
    float* d_x, float* d_y,
    float* d_destX, float* d_destY, float* d_destR,
    int* d_wpIndex, const int* d_wpCount, const int* d_wpOffset,
    const float* d_wpPoolX, const float* d_wpPoolY, const float* d_wpPoolR,
    int numAgents)
{
    const int blockSizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);
    const int iters = 10000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("ThreadsPerBlock | Avg kernel time (ms)\n");
    printf("-------------------------------------\n");

    for (int i = 0; i < numTests; i++) {
        int threadsPerBlock = blockSizes[i];
        int numBlocks =
            (numAgents + (threadsPerBlock * 4) - 1) / (threadsPerBlock * 4);

        // Warm-up
        for (int w = 0; w < 10; w++) {
            cudaMoveKernel<<<numBlocks, threadsPerBlock>>>(
                d_x, d_y, d_destX, d_destY, d_destR,
                d_wpIndex, d_wpCount, d_wpOffset,
                d_wpPoolX, d_wpPoolY, d_wpPoolR,
                numAgents);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int it = 0; it < iters; it++) {
            cudaMoveKernel<<<numBlocks, threadsPerBlock>>>(
                d_x, d_y, d_destX, d_destY, d_destR,
                d_wpIndex, d_wpCount, d_wpOffset,
                d_wpPoolX, d_wpPoolY, d_wpPoolR,
                numAgents);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("%15d | %8.4f\n",
               threadsPerBlock, ms / iters);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int numAgents = 1 << 20; // ~1 million agents

    float *d_x, *d_y, *d_destX, *d_destY, *d_destR;
    int *d_wpIndex, *d_wpCount, *d_wpOffset;
    float *d_wpPoolX, *d_wpPoolY, *d_wpPoolR;

    cudaMalloc(&d_x, numAgents * sizeof(float));
    cudaMalloc(&d_y, numAgents * sizeof(float));
    cudaMalloc(&d_destX, numAgents * sizeof(float));
    cudaMalloc(&d_destY, numAgents * sizeof(float));
    cudaMalloc(&d_destR, numAgents * sizeof(float));

    cudaMalloc(&d_wpIndex, numAgents * sizeof(int));
    cudaMalloc(&d_wpCount, numAgents * sizeof(int));
    cudaMalloc(&d_wpOffset, numAgents * sizeof(int));

    cudaMalloc(&d_wpPoolX, numAgents * sizeof(float));
    cudaMalloc(&d_wpPoolY, numAgents * sizeof(float));
    cudaMalloc(&d_wpPoolR, numAgents * sizeof(float));

    testBlockSizes(
        d_x, d_y,
        d_destX, d_destY, d_destR,
        d_wpIndex, d_wpCount, d_wpOffset,
        d_wpPoolX, d_wpPoolY, d_wpPoolR,
        numAgents
    );

    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_destX);
    cudaFree(d_destY);
    cudaFree(d_destR);
    cudaFree(d_wpIndex);
    cudaFree(d_wpCount);
    cudaFree(d_wpOffset);
    cudaFree(d_wpPoolX);
    cudaFree(d_wpPoolY);
    cudaFree(d_wpPoolR);

    return 0;
}
