#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Process 4 agents per thread for better GPU utilization
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
    int numAgents) 
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 agents in registers
    #pragma unroll
    for (int lane = 0; lane < 4; lane++) {
        int i = idx + lane;
        if (i >= numAgents) return;
        
        // Load to registers (coalesced)
        float px = x[i];
        float py = y[i];
        float dx = destX[i];
        float dy = destY[i];
        float dr = destR[i];
        int curr = wpIndex[i];
        int count = wpCount[i];
        int off = wpOffset[i];
        
        // Calculate movement direction
        float diffX = dx - px;
        float diffY = dy - py;
        float distSq = fmaf(diffX, diffX, diffY * diffY); // FMA: diffX*diffX + diffY*diffY
        
        // Move toward destination (unit step)
        if (distSq > 1e-10f) {
            float invDist = rsqrtf(distSq);  // Fast 1/sqrt
            px = fmaf(diffX, invDist, px);   // FMA: px + diffX * invDist
            py = fmaf(diffY, invDist, py);
        }
        
        // Check if reached destination (NEW position, squared distance)
        diffX = dx - px;
        diffY = dy - py;
        float newDistSq = fmaf(diffX, diffX, diffY * diffY);
        
        // Update waypoint if reached
        if (newDistSq < (dr * dr)) {
            if (count > 0) {
                curr++;
                if (curr >= count) curr = 0;  // Faster than modulo
                
                int poolIdx = off + curr;
                dx = wpPoolX[poolIdx];
                dy = wpPoolY[poolIdx];
                dr = wpPoolR[poolIdx];
            }
        }
        
        // Write back (coalesced)
        x[i] = px;
        y[i] = py;
        destX[i] = dx;
        destY[i] = dy;
        destR[i] = dr;
        wpIndex[i] = curr;
    }
}

extern "C" void cudaKernelfunction(
    float* d_x, float* d_y, 
    float* d_destX, float* d_destY, float* d_destR,
    int* d_wpIndex, const int* d_wpCount, const int* d_wpOffset,
    const float* d_wpPoolX, const float* d_wpPoolY, const float* d_wpPoolR,
    int numAgents, cudaStream_t stream) 
{
    // Each thread processes 4 agents
    int threadsPerBlock = 16;
    int numBlocks = (numAgents + (threadsPerBlock * 4) - 1) / (threadsPerBlock * 4);
    
    cudaMoveKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_x, d_y, d_destX, d_destY, d_destR,
        d_wpIndex, d_wpCount, d_wpOffset,
        d_wpPoolX, d_wpPoolY, d_wpPoolR,
        numAgents);
}