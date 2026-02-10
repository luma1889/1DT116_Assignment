#include <cuda_runtime.h>

// Optimized CUDA kernel with 8 agents per thread
__global__ void cudaMoveKernel(float* x, float* y, 
                               float* destX, float* destY, float* destR,
                               int* wpIndex, const int* wpCount, const int* wpOffset,
                               const float* wpPoolX, const float* wpPoolY, const float* wpPoolR,
                               int numAgents) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx < numAgents) {
        // Process 8 agents in parallel using registers
        float px[8], py[8], dx[8], dy[8], dr[8];
        int idxArr[8], cnt[8], off[8], curr[8];
        
        // Coalesced reads
        for (int i = 0; i < 8 && (idx + i) < numAgents; i++) {
            int agentIdx = idx + i;
            px[i] = x[agentIdx];
            py[i] = y[agentIdx];
            dx[i] = destX[agentIdx];
            dy[i] = destY[agentIdx];
            dr[i] = destR[agentIdx];
            idxArr[i] = wpIndex[agentIdx];
            cnt[i] = wpCount[agentIdx];
            off[i] = wpOffset[agentIdx];
            curr[i] = idxArr[i];
        }
        
        // Process all 8 agents
        for (int i = 0; i < 8 && (idx + i) < numAgents; i++) {
            // Calculate movement
            float diffX = dx[i] - px[i];
            float diffY = dy[i] - py[i];
            float distSq = diffX * diffX + diffY * diffY;
            
            if (distSq > 1e-10f) {
                float invDist = rsqrtf(distSq);
                px[i] += diffX * invDist;
                py[i] += diffY * invDist;
            }
            
            // Check destination reached
            float newDiffX = dx[i] - px[i];
            float newDiffY = dy[i] - py[i];
            if ((newDiffX * newDiffX + newDiffY * newDiffY) < (dr[i] * dr[i])) {
                curr[i] = (curr[i] + 1) % cnt[i];
                
                // Update destination from pool
                int poolIdx = off[i] + curr[i];
                dx[i] = wpPoolX[poolIdx];
                dy[i] = wpPoolY[poolIdx];
                dr[i] = wpPoolR[poolIdx];
            }
        }
        
        // Coalesced writes
        for (int i = 0; i < 8 && (idx + i) < numAgents; i++) {
            int agentIdx = idx + i;
            x[agentIdx] = px[i];
            y[agentIdx] = py[i];
            destX[agentIdx] = dx[i];
            destY[agentIdx] = dy[i];
            destR[agentIdx] = dr[i];
            wpIndex[agentIdx] = curr[i];
        }
    }
}

extern "C" void cudaKernelfunction(float* d_x, float* d_y, 
                                   float* d_destX, float* d_destY, float* d_destR,
                                   int* d_wpIndex, const int* d_wpCount, const int* d_wpOffset,
                                   const float* d_wpPoolX, const float* d_wpPoolY, const float* d_wpPoolR,
                                   int numAgents, cudaStream_t stream) 
{
    // Each thread handles 8 agents
    int threadsPerBlock = 256;
    int agentsPerBlock = threadsPerBlock * 8;
    int blocks = (numAgents + agentsPerBlock - 1) / agentsPerBlock;
    
    cudaMoveKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        d_x, d_y, d_destX, d_destY, d_destR,
        d_wpIndex, d_wpCount, d_wpOffset,
        d_wpPoolX, d_wpPoolY, d_wpPoolR,
        numAgents);
}