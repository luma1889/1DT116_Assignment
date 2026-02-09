#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <cstdint>
#include "ped_model.h"

__global__ void cudaMove(float *agentX, float *agentY, 
                         const float *destX, const float *destY, const float *destR, int numAgents)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numAgents)
    {
        float dx = destX[i] - agentX[i];
        float dy = destY[i] - agentY[i];
        float sq_len = dx * dx + dy * dy;

        if (sq_len > 1e-7f)
        {
            float invLen = rsqrtf(sq_len);
            agentX[i] += dx * invLen;
            agentY[i] += dy * invLen;
        }

        float new_dx = destX[i] - agentX[i];
        float new_dy = destY[i] - agentY[i];
    }
}

extern "C" void cudaKernelfunction(float *d_agentX, float *d_agentY, 
                                   float *d_destX, float *d_destY, float *d_destR, int numAgents)
{

    int blockSize = 256;
    int numBlocks = (numAgents + blockSize - 1) / blockSize;

    cudaMove<<<numBlocks, blockSize>>>(d_agentX, d_agentY, d_destX, d_destY, d_destR, numAgents);
}