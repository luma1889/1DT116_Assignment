#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <cstdint>
#include "ped_model.h"

// Inside your .cu file
__global__ void cudaMove(float *agentX, float *agentY, 
                         const float *destX, const float *destY, const float *destR, 
                         int8_t *reached, int numAgents) // Added reached here
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
            agentX[i] = fmaf(dx, invLen, agentX[i]);
            agentY[i] = fmaf(dy, invLen, agentY[i]);
        }

        // Calculate if reached AFTER moving
        float nx = destX[i] - agentX[i];
        float ny = destY[i] - agentY[i];
        reached[i] = ((nx * nx + ny * ny) < (destR[i] * destR[i])) ? 1 : 0;
    }
}

extern "C" void cudaKernelfunction(float *d_agentX, float *d_agentY, 
                                   float *d_destX, float *d_destY, float *d_destR,
                                   int8_t *d_reached, int numAgents) // Match signature
{
    // Remove the semicolon that was after the function header in your snippet!
    int blockSize = 256;
    int numBlocks = (numAgents + blockSize - 1) / blockSize;

    cudaMove<<<numBlocks, blockSize>>>(d_agentX, d_agentY, d_destX, d_destY, d_destR, d_reached, numAgents);
}