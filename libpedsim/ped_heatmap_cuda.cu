//
// pedsim - CUDA heatmap kernels for Assignment 4
//
// Three parallelised heatmap stages:
//   1. Fade        – multiply every cell by 0.80  (one thread per cell)
//   2. Increment   – atomicAdd 40 at each desired position (one thread per agent)
//   3. Clamp       – cap at 255                   (one thread per cell)
//   4. Scale       – nearest-neighbour upscale     (one thread per output pixel)
//   5. Blur        – 5×5 Gaussian, shared-memory tiles (one thread per output pixel)
//
// All kernels are issued to the same cudaStream, so they execute in order on the
// GPU while the CPU is free to run collision handling concurrently.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ── Compile-time constants (must match the values in ped_model.h) ─────────────
#define HMAP_SIZE       160
#define HMAP_CELLSIZE   5
#define HMAP_SCALED     (HMAP_SIZE * HMAP_CELLSIZE)   // 800

// ── Blur kernel tile configuration ────────────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 1 – Fade
// Each thread handles one cell of the flat heatmap array.
// Memory access: stride-1 read + write → perfectly coalesced.
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void fadeKernel(int* __restrict__ hm, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        hm[i] = __float2int_rn(hm[i] * 0.80f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 2 – Increment
// Each thread handles one agent; increments the desired-position cell by 40
// using atomicAdd to avoid data races when multiple agents target the same cell.
// Reads of desiredX/Y are stride-1 (one float per agent) → coalesced.
// Writes are scattered (depends on agent positions) → inherently non-coalesced,
// but unavoidable for this algorithm.
// ═══════════════════════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 3 – Clamp
// Cap every cell at 255 so colour saturation is not exceeded.
// Memory access: stride-1 → coalesced.
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void clampKernel(int* __restrict__ hm, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        hm[i] = min(hm[i], 255);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 4 – Scale
// Each thread writes one pixel of the HMAP_SCALED × HMAP_SCALED output grid.
// Reads from d_heatmap are broadcast (HMAP_CELLSIZE consecutive threads share
// the same source cell) → served from L1 cache after the first access.
// Writes to d_scaled are stride-1 → coalesced.
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void scaleKernel(const int* __restrict__ hm,
                             int* __restrict__       scaled)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    int sy = blockIdx.y * blockDim.y + threadIdx.y;  // output row

    if (sx < HMAP_SCALED && sy < HMAP_SCALED)
        scaled[sy * HMAP_SCALED + sx] =
            hm[(sy / HMAP_CELLSIZE) * HMAP_SIZE + (sx / HMAP_CELLSIZE)];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel 5 – Gaussian blur (5×5) with shared-memory tiles
//
// Each BLUR_TILE × BLUR_TILE thread-block computes BLUR_TILE × BLUR_TILE output
// pixels.  Before computing, it cooperatively loads a (BLUR_TILE + 2*RADIUS) ×
// (BLUR_TILE + 2*RADIUS) = 36×36 halo patch into shared memory.
//
// Memory access pattern:
//   Load phase  – threads load elements in row-major order.  Within a warp all
//                 32 threads access 32 consecutive columns in the same row of
//                 the input array → perfectly coalesced on the first pass, and
//                 nearly coalesced on the second (one row-boundary crossing).
//   Compute phase – all reads come from shared memory → ~10× lower latency than
//                   global memory, and ~10× higher bandwidth.
//   Store phase – each thread writes one int to a stride-1 location → coalesced.
//
// Grid size: ceil(5120/32) × ceil(5120/32) = 160 × 160 = 25 600 blocks, well
// above the "more than one block" requirement.
//
// Shared memory per block: 36 × 36 × 4 B = 5 184 B  (well within the 48 KB
// per-SM limit on all supported devices).
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void blurKernel(const int* __restrict__ in,
                            int* __restrict__       out)
{
    __shared__ int smem[SMEM_DIM][SMEM_DIM];

    const int W = HMAP_SCALED;

    // Top-left corner of the halo patch in global coordinates
    const int origX = (int)(blockIdx.x * BLUR_TILE) - BLUR_RADIUS;
    const int origY = (int)(blockIdx.y * BLUR_TILE) - BLUR_RADIUS;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * BLUR_TILE + tx;

    // ── Cooperative halo load ─────────────────────────────────────────────────
    // We need SMEM_DIM * SMEM_DIM = 1296 elements; the block has 1024 threads.
    // Each thread handles up to 2 elements via a stride-equal-to-block-size loop.
    // Within a warp, thread i handles element index tid, so consecutive threads
    // load consecutive memory locations (same row, adjacent columns) → coalesced.
    for (int p = tid; p < SMEM_DIM * SMEM_DIM; p += BLUR_TILE * BLUR_TILE) {
        int sy = p / SMEM_DIM;
        int sx = p % SMEM_DIM;
        int gy = origY + sy;
        int gx = origX + sx;
        smem[sy][sx] = (gx >= 0 && gx < W && gy >= 0 && gy < W)
                       ? in[gy * W + gx] : 0;
    }
    __syncthreads();

    // ── Compute blur for this thread's output pixel ───────────────────────────
    int outX = blockIdx.x * BLUR_TILE + tx;
    int outY = blockIdx.y * BLUR_TILE + ty;

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
        
        // Manually blend White to Red (No transparency needed!)
        int gb = 255 - val;
        out[outY * W + outX] = 0xFF000000 | (0xFF << 16) | (gb << 8) | gb;
    } 
    else if (outX < W && outY < W) 
    {
        // Force the border pixels to be solid White instead of transparent
        out[outY * W + outX] = 0xFFFFFFFF;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host-side launcher — called from ped_model.cpp
//
// Issues all five kernels to 'stream'.  The call returns immediately; the GPU
// work proceeds asynchronously so the CPU can run collision handling in parallel.
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" void launchHeatmapKernels(
    cudaStream_t  stream,
    int*          d_heatmap,
    int*          d_scaled,
    int*          d_blurred,
    float*        d_desiredX,
    float*        d_desiredY,
    const float*  h_desiredX,   // host arrays (agentData.desiredX / desiredY)
    const float*  h_desiredY,
    int           numAgents)
{
    // ── Step 0: copy desired positions to GPU (async, same stream) ────────────
    cudaMemcpyAsync(d_desiredX, h_desiredX, numAgents * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_desiredY, h_desiredY, numAgents * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    const int THR = 256;

    // ── Step 1: Fade ──────────────────────────────────────────────────────────
    {
        int total = HMAP_SIZE * HMAP_SIZE;
        fadeKernel<<<(total + THR - 1) / THR, THR, 0, stream>>>(d_heatmap, total);
    }

    // ── Step 2: Increment ─────────────────────────────────────────────────────
    {
        incrementKernel<<<(numAgents + THR - 1) / THR, THR, 0, stream>>>(
            d_heatmap, d_desiredX, d_desiredY, numAgents);
    }

    // ── Step 3: Clamp ─────────────────────────────────────────────────────────
    {
        int total = HMAP_SIZE * HMAP_SIZE;
        clampKernel<<<(total + THR - 1) / THR, THR, 0, stream>>>(d_heatmap, total);
    }

    // ── Step 4: Scale ─────────────────────────────────────────────────────────
    {
        dim3 block(16, 16);
        dim3 grid((HMAP_SCALED + 15) / 16, (HMAP_SCALED + 15) / 16);
        scaleKernel<<<grid, block, 0, stream>>>(d_heatmap, d_scaled);
    }

    // ── Step 5: Blur ──────────────────────────────────────────────────────────
    {
        dim3 block(BLUR_TILE, BLUR_TILE);
        dim3 grid((HMAP_SCALED + BLUR_TILE - 1) / BLUR_TILE,
                  (HMAP_SCALED + BLUR_TILE - 1) / BLUR_TILE);
        blurKernel<<<grid, block, 0, stream>>>(d_scaled, d_blurred);
    }
}
