// pedsim - CUDA heatmap kernels for Assignment 4
//
// Three parallelised heatmap stages:
//   1. Fade        – multiply every cell by 0.80  (one thread per cell)
//   2. Increment   – atomicAdd 40 at each desired position (one thread per agent)
//   3. Clamp       – cap at 255                   (one thread per cell)
//   4. Scale       – nearest-neighbour upscale     (one thread per output pixel)
//   5. Blur        – 5×5 Gaussian, shared-memory tiles (one thread per output pixel)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define HMAP_SIZE       160
#define HMAP_CELLSIZE   5
#define HMAP_SCALED     (HMAP_SIZE * HMAP_CELLSIZE)   // 800

#define BLUR_TILE       32
#define BLUR_RADIUS     2
#define SMEM_DIM        (BLUR_TILE + 2 * BLUR_RADIUS)  // 36
#define WEIGHTSUM       273

// 5×5 Gaussian weights stored in fast constant memory
__constant__ int gaussW[5][5] = {
    { 1,  4,  7,  4,  1 },
    { 4, 16, 26, 16,  4 },
    { 7, 26, 41, 26,  7 },
    { 4, 16, 26, 16,  4 },
    { 1,  4,  7,  4,  1 }
};

// Kernel 1 – Fade
__global__ void fadeKernel(int* __restrict__ hm, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        hm[i] = __float2int_rn(hm[i] * 0.80f);
}

// Kernel 2 – Increment
__global__ void incrementKernel(int* __restrict__ hm,
                                  const float* __restrict__ desX,
                                  const float* __restrict__ desY,
                                  int numAgents)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAgents) return;

    int x = __float2int_rn(desX[i]);
    int y = __float2int_rn(desY[i]);
    if (x >= 0 && x < HMAP_SIZE && y >= 0 && y < HMAP_SIZE)
        atomicAdd(&hm[y * HMAP_SIZE + x], 40);
}

// Kernel 3 – Clamp
__global__ void clampKernel(int* __restrict__ hm, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        hm[i] = min(hm[i], 255);
}

// Kernel 4 – Scale
__global__ void scaleKernel(const int* __restrict__ hm,
                             int* __restrict__       scaled)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx < HMAP_SCALED && sy < HMAP_SCALED)
        scaled[sy * HMAP_SCALED + sx] =
            hm[(sy / HMAP_CELLSIZE) * HMAP_SIZE + (sx / HMAP_CELLSIZE)];
}

// Kernel 5 – Gaussian blur (5×5) with shared-memory
__global__ void blurKernel(const int* __restrict__ in,
                            int* __restrict__       out)
{
    __shared__ int smem[SMEM_DIM][SMEM_DIM];

    const int W = HMAP_SCALED;

    // top left corner
    const int origX = (int)(blockIdx.x * BLUR_TILE) - BLUR_RADIUS;
    const int origY = (int)(blockIdx.y * BLUR_TILE) - BLUR_RADIUS;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * BLUR_TILE + tx;

    for (int p = tid; p < SMEM_DIM * SMEM_DIM; p += BLUR_TILE * BLUR_TILE) {
        int sy = p / SMEM_DIM;
        int sx = p % SMEM_DIM;
        int gy = origY + sy;
        int gx = origX + sx;
        smem[sy][sx] = (gx >= 0 && gx < W && gy >= 0 && gy < W)
                       ? in[gy * W + gx] : 0;
    }
    __syncthreads();

    // Output pixel coordinates
    int outX = blockIdx.x * BLUR_TILE + tx;
    int outY = blockIdx.y * BLUR_TILE + ty;

    // Only compute blur for pixels that have a full 5×5
    if (outX >= BLUR_RADIUS && outX < W - BLUR_RADIUS &&
        outY >= BLUR_RADIUS && outY < W - BLUR_RADIUS)
    {
        int sum = 0;
        #pragma unroll
        for (int dy = 0; dy < 5; dy++)
            #pragma unroll
            for (int dx = 0; dx < 5; dx++)
                sum += gaussW[dy][dx] * smem[ty + dy][tx + dx];

        int val = sum / WEIGHTSUM;
        val = min(val, 255); // Prevent color overflow
        
        // Manually blend White to Red
        int gb = 255 - val;
        out[outY * W + outX] = 0xFF000000 | (0xFF << 16) | (gb << 8) | gb;
    } 
    else if (outX < W && outY < W) 
    {
        // Force the border pixels to be solid White
        out[outY * W + outX] = 0xFFFFFFFF;
    }
}

static cudaEvent_t ev_start, ev_fade, ev_inc, ev_scale, ev_blur;
static bool ev_init = false;
static float total_fade = 0, total_inc = 0, total_scale = 0, total_blur = 0;
static int gpu_frames = 0;

// Host function to launch all heatmap kernels in sequence
extern "C" void launchHeatmapKernels(
    cudaStream_t stream, int* d_heatmap, int* d_scaled, int* d_blurred,
    float* d_desiredX, float* d_desiredY, const float* h_desiredX, const float* h_desiredY, int numAgents)
{
    if (!ev_init) {
        cudaEventCreate(&ev_start); cudaEventCreate(&ev_fade);
        cudaEventCreate(&ev_inc);   cudaEventCreate(&ev_scale);
        cudaEventCreate(&ev_blur);
        ev_init = true;
    }

    // Step 0: Data transfer
    cudaMemcpyAsync(d_desiredX, h_desiredX, numAgents * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_desiredY, h_desiredY, numAgents * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaEventRecord(ev_start, stream);

    // Step 1: Fade
    int total = HMAP_SIZE * HMAP_SIZE;
    const int THR = 256;
    fadeKernel<<<(total + THR - 1) / THR, THR, 0, stream>>>(d_heatmap, total);
    
    cudaEventRecord(ev_fade, stream);

    // Step 2: Increment & Clamp
    incrementKernel<<<(numAgents + THR - 1) / THR, THR, 0, stream>>>(d_heatmap, d_desiredX, d_desiredY, numAgents);
    clampKernel<<<(total + THR - 1) / THR, THR, 0, stream>>>(d_heatmap, total);
    
    cudaEventRecord(ev_inc, stream);

    // Step 3: Scale
    dim3 scaleBlock(16, 16);
    dim3 scaleGrid((HMAP_SCALED + 15) / 16, (HMAP_SCALED + 15) / 16);
    scaleKernel<<<scaleGrid, scaleBlock, 0, stream>>>(d_heatmap, d_scaled);
    
    cudaEventRecord(ev_scale, stream);

    // Step 4: Blur
    dim3 blurBlock(BLUR_TILE, BLUR_TILE);
    dim3 blurGrid((HMAP_SCALED + BLUR_TILE - 1) / BLUR_TILE, (HMAP_SCALED + BLUR_TILE - 1) / BLUR_TILE);
    blurKernel<<<blurGrid, blurBlock, 0, stream>>>(d_scaled, d_blurred);
    
    cudaEventRecord(ev_blur, stream);
}

// Host function to calculate and accumulate timings for each kernel
extern "C" void calculateHeatmapTimings()
{
    // failsafe, if we have nothing to time
    if (!ev_init) return;

    float ms_fade = 0, ms_inc = 0, ms_scale = 0, ms_blur = 0;
    
    // Calculate the difference between recorded events
    cudaEventElapsedTime(&ms_fade,  ev_start, ev_fade);
    cudaEventElapsedTime(&ms_inc,   ev_fade,  ev_inc);
    cudaEventElapsedTime(&ms_scale, ev_inc,   ev_scale);
    cudaEventElapsedTime(&ms_blur,  ev_scale, ev_blur);

    // Accumulate totals for the average
    total_fade += ms_fade;
    total_inc  += ms_inc;
    total_scale += ms_scale;
    total_blur += ms_blur;
    
    gpu_frames++;
}

// Called automatically when the program exits
extern "C" void printGPUHeatmapAverages()
{
    if (gpu_frames > 0) {
        printf("\n================ GPU TIMING RESULTS ================\n");
        printf("Average GPU Fade:      %.4f ms\n", total_fade / gpu_frames);
        printf("Average GPU Increment: %.4f ms\n", total_inc / gpu_frames);
        printf("Average GPU Scale:     %.4f ms\n", total_scale / gpu_frames);
        printf("Average GPU Blur:      %.4f ms\n", total_blur / gpu_frames);
        printf("Average TOTAL GPU:     %.4f ms\n", (total_fade + total_inc + total_scale + total_blur) / gpu_frames);
        printf("====================================================\n\n");
    }
}