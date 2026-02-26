//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <thread>
#include <omp.h>
#include <cstdlib>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
// Declare CUDA kernel function
extern "C" void cudaKernelfunction(float *d_x, float *d_y,
                                   float *d_destX, float *d_destY, float *d_destR,
                                   int *d_wpIndex, const int *d_wpCount, const int *d_wpOffset,
                                   const float *d_wpPoolX, const float *d_wpPoolY, const float *d_wpPoolR,
                                   int numAgents, cudaStream_t stream);
#endif

// Alignment for AVX2 (32 bytes)
#define ALIGNMENT 32
#define AGENTS_PER_VECTOR 8

template <typename T>
T *aligned_alloc(size_t count)
{
    return static_cast<T *>(_mm_malloc(count * sizeof(T), ALIGNMENT));
}

void Ped::Model::setup(std::vector<Tagent *> agentsInScenario,
                       std::vector<Twaypoint *> destinationsInScenario,
                       IMPLEMENTATION implementation)
{
    this->implementation = implementation;
    this->agents = agentsInScenario;
    this->destinations = destinationsInScenario;
    this->isCleaned = false;

    // Setup ID-based architecture
    for (int i = 0; i < (int)agents.size(); ++i)
    {
        agents[i]->setId(i, this);
    }

    stuckCounter.resize(agents.size(), 0);

    // Allocate and initialize arrays
    allocateArrays();
    buildWaypointPool();
    initializeArrays();

    // Initialise the collision board
    board = new std::atomic<int>[WORLD_WIDTH * WORLD_HEIGHT];
    for (int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; ++i)
        board[i].store(-1, std::memory_order_relaxed);

    for (int i = 0; i < agentData.count; ++i) {
        int x = (int)agentData.x[i];
        int y = (int)agentData.y[i];
        if (x >= 0 && x < WORLD_WIDTH && y >= 0 && y < WORLD_HEIGHT)
            board[y * WORLD_WIDTH + x].store(i, std::memory_order_relaxed);
    }

    if (implementation == REGION) {
        initRegions();
    }

    // CUDA-specific setup
    if (implementation == CUDA)
    {
#ifdef USE_CUDA
        setupCUDA();
#endif
    }

    // setupHeatmapSeq();
}

void Ped::Model::allocateArrays()
{
    agentData.count = agents.size();
    agentData.paddedCount = ((agentData.count + AGENTS_PER_VECTOR - 1) / AGENTS_PER_VECTOR) * AGENTS_PER_VECTOR;

    // Allocate aligned arrays
    agentData.x = aligned_alloc<float>(agentData.paddedCount);
    agentData.y = aligned_alloc<float>(agentData.paddedCount);
    agentData.destX = aligned_alloc<float>(agentData.paddedCount);
    agentData.destY = aligned_alloc<float>(agentData.paddedCount);
    agentData.destR = aligned_alloc<float>(agentData.paddedCount);
    agentData.wpIndex = aligned_alloc<int>(agentData.paddedCount);
    agentData.wpCount = aligned_alloc<int>(agentData.paddedCount);
    agentData.wpOffset = aligned_alloc<int>(agentData.paddedCount);

    agentData.desiredX = aligned_alloc<float>(agentData.paddedCount);
    agentData.desiredY = aligned_alloc<float>(agentData.paddedCount);

    // Initialize all pointers to nullptr
    agentData.wpPoolX = nullptr;
    agentData.wpPoolY = nullptr;
    agentData.wpPoolR = nullptr;
}

void Ped::Model::buildWaypointPool()
{
    // Count total waypoints
    int totalWaypoints = 0;
    for (auto agent : agents)
    {
        totalWaypoints += std::max(1, (int)agent->getWaypoints().size());
    }

    agentData.wpPoolSize = totalWaypoints;

    // Allocate waypoint pool
    agentData.wpPoolX = aligned_alloc<float>(totalWaypoints);
    agentData.wpPoolY = aligned_alloc<float>(totalWaypoints);
    agentData.wpPoolR = aligned_alloc<float>(totalWaypoints);

    // Fill waypoint pool
    int offset = 0;
    for (int i = 0; i < agentData.count; ++i)
    {
        agentData.wpOffset[i] = offset;
        agentData.wpIndex[i] = 0;

        const auto &waypoints = agents[i]->getWaypoints();
        agentData.wpCount[i] = std::max(1, (int)waypoints.size());

        if (waypoints.empty())
        {
            // No waypoints - stay in place
            agentData.wpPoolX[offset] = (float)agents[i]->getInitX();
            agentData.wpPoolY[offset] = (float)agents[i]->getInitY();
            agentData.wpPoolR[offset] = 1.0f;
            offset++;
        }
        else
        {
            for (auto wp : waypoints)
            {
                agentData.wpPoolX[offset] = (float)wp->getx();
                agentData.wpPoolY[offset] = (float)wp->gety();
                agentData.wpPoolR[offset] = (float)wp->getr();
                offset++;
            }
        }
    }
}

void Ped::Model::initializeArrays()
{
    // Initialize arrays with agent data
    for (int i = 0; i < agentData.count; ++i)
    {
        agentData.x[i] = (float)agents[i]->getInitX();
        agentData.y[i] = (float)agents[i]->getInitY();
        agentData.desiredX[i] = agentData.x[i];
        agentData.desiredY[i] = agentData.y[i];

        // Set initial destination
        int wpIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
        agentData.destX[i] = agentData.wpPoolX[wpIdx];
        agentData.destY[i] = agentData.wpPoolY[wpIdx];
        agentData.destR[i] = agentData.wpPoolR[wpIdx];
    }

    // Pad remaining elements
    for (int i = agentData.count; i < agentData.paddedCount; ++i)
    {
        agentData.x[i] = 0.0f;
        agentData.y[i] = 0.0f;
        agentData.destX[i] = 0.0f;
        agentData.destY[i] = 0.0f;
        agentData.destR[i] = 1.0f;
        agentData.wpIndex[i] = 0;
        agentData.wpCount[i] = 1;
        agentData.wpOffset[i] = 0;
        agentData.desiredX[i] = 0.0f;
        agentData.desiredY[i] = 0.0f;
    }
}

void Ped::Model::tick()
{
    switch (implementation)
    {
    case SEQ:
        tickSEQ();
        break;
    case OMP:
        tickOMP();
        break;
    case PTHREAD:
        tickPTHREAD();
        break;
    case VECTOR:
        tickVECTOR();
        break;
    case CUDA:
        tickCUDA();
        break;
    case REGION:
        tickREGION();
        break;
    }
    // updateHeatmapSeq();
}

// Optimized sequential implementation
void Ped::Model::tickSEQ()
{
    // Compute DESIRED positions and waypoints
    for (int i = 0; i < agentData.count; ++i)
    {
        float dx = agentData.destX[i] - agentData.x[i];
        float dy = agentData.destY[i] - agentData.y[i];
        float distSq = dx * dx + dy * dy;
        
        // Update waypoint if reached
        if (distSq < agentData.destR[i] * agentData.destR[i]) {
            if (agentData.wpCount[i] > 0) {
                int next = agentData.wpIndex[i] + 1;
                if (next >= agentData.wpCount[i]) next = 0;
                agentData.wpIndex[i] = next;
                
                int poolIdx = agentData.wpOffset[i] + next;
                agentData.destX[i] = agentData.wpPoolX[poolIdx];
                agentData.destY[i] = agentData.wpPoolY[poolIdx];
                agentData.destR[i] = agentData.wpPoolR[poolIdx];
                
                // Recalculate dx/dy
                dx = agentData.destX[i] - agentData.x[i];
                dy = agentData.destY[i] - agentData.y[i];
                distSq = dx * dx + dy * dy;
            }
        }
        
        // Math Vector
        if (distSq > 1e-10f) {
            float invDist = 1.0f / sqrtf(distSq);
            agentData.desiredX[i] = agentData.x[i] + dx * invDist;
            agentData.desiredY[i] = agentData.y[i] + dy * invDist;
        } else {
            agentData.desiredX[i] = agentData.x[i];
            agentData.desiredY[i] = agentData.y[i];
        }
    }
    
    // Move
    for (int i = 0; i < agentData.count; ++i) {
        move(i);
    }
}

// Optimized OpenMP implementation
void Ped::Model::tickOMP() 
{
    // MATH (Calculate where agents want to go)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < agentData.count; ++i) {
        float dx = agentData.destX[i] - agentData.x[i];
        float dy = agentData.destY[i] - agentData.y[i];
        float distSq = dx * dx + dy * dy;
        float radSq = agentData.destR[i] * agentData.destR[i];
        
        // Waypoint Logic
        if (distSq < radSq && agentData.wpCount[i] > 0) {
            agentData.wpIndex[i] = (agentData.wpIndex[i] + 1) % agentData.wpCount[i];
            int poolIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
            agentData.destX[i] = agentData.wpPoolX[poolIdx];
            agentData.destY[i] = agentData.wpPoolY[poolIdx];
            agentData.destR[i] = agentData.wpPoolR[poolIdx];
            dx = agentData.destX[i] - agentData.x[i];
            dy = agentData.destY[i] - agentData.y[i];
            distSq = dx*dx + dy*dy;
        }
        
        // Math Vector
        if (distSq > 1e-10f) {
            float invDist = 1.0f / sqrtf(distSq);
            agentData.desiredX[i] = agentData.x[i] + dx * invDist;
            agentData.desiredY[i] = agentData.y[i] + dy * invDist;
        } else {
            agentData.desiredX[i] = agentData.x[i];
            agentData.desiredY[i] = agentData.y[i];
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < agentData.count; ++i) {
        move(i);
    }
}

// PThread implementation
void Ped::Model::tickPTHREAD()
{
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (agentData.count + num_threads - 1) / num_threads;

    auto worker = [this](int start, int end)
    {
        for (int i = start; i < end; ++i)
        {
            float dx = agentData.destX[i] - agentData.x[i];
            float dy = agentData.destY[i] - agentData.y[i];
            float distSq = dx * dx + dy * dy;
            
            if (distSq < agentData.destR[i] * agentData.destR[i]) {
                if (agentData.wpCount[i] > 0) {
                    int next = agentData.wpIndex[i] + 1;
                    if (next >= agentData.wpCount[i]) next = 0;
                    agentData.wpIndex[i] = next;
                    
                    int poolIdx = agentData.wpOffset[i] + next;
                    agentData.destX[i] = agentData.wpPoolX[poolIdx];
                    agentData.destY[i] = agentData.wpPoolY[poolIdx];
                    agentData.destR[i] = agentData.wpPoolR[poolIdx];
                    
                    dx = agentData.destX[i] - agentData.x[i];
                    dy = agentData.destY[i] - agentData.y[i];
                }
            }
            
            float distSqNew = dx * dx + dy * dy;
            if (distSqNew > 1e-10f) {
                float invDist = 1.0f / sqrtf(distSqNew);
                agentData.x[i] += dx * invDist;
                agentData.y[i] += dy * invDist;
            }
        }
    };

    for (int t = 0; t < num_threads; ++t)
    {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, agentData.count);
        if (start < end)
        {
            threads.emplace_back(worker, start, end);
        }
    }

    for (auto &t : threads)
        t.join();
}

// OPTIMIZED VECTOR IMPLEMENTATION (Fully Vectorized)
void Ped::Model::tickVECTOR()
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < agentData.paddedCount; i += AGENTS_PER_VECTOR)
    {
        // Load positions and destinations
        __m256 posX = _mm256_load_ps(&agentData.x[i]);
        __m256 posY = _mm256_load_ps(&agentData.y[i]);
        __m256 destX = _mm256_load_ps(&agentData.destX[i]);
        __m256 destY = _mm256_load_ps(&agentData.destY[i]);
        __m256 destR = _mm256_load_ps(&agentData.destR[i]);

        // Calculate direction vectors
        __m256 diffX = _mm256_sub_ps(destX, posX);
        __m256 diffY = _mm256_sub_ps(destY, posY);

        // Calculate squared distance using FMA
        __m256 distSq = _mm256_fmadd_ps(diffY, diffY, _mm256_mul_ps(diffX, diffX));

        // Calculate normalized movement
        __m256 mask = _mm256_cmp_ps(distSq, _mm256_set1_ps(1e-10f), _CMP_GT_OQ);
        __m256 invDist = _mm256_rsqrt_ps(distSq);
        invDist = _mm256_and_ps(invDist, mask);

        // Update positions using FMA
        posX = _mm256_fmadd_ps(diffX, invDist, posX);
        posY = _mm256_fmadd_ps(diffY, invDist, posY);

        // Store updated positions
        _mm256_store_ps(&agentData.x[i], posX);
        _mm256_store_ps(&agentData.y[i], posY);

        // Check if destinations reached
        __m256 newDiffX = _mm256_sub_ps(destX, posX);
        __m256 newDiffY = _mm256_sub_ps(destY, posY);
        __m256 newDistSq = _mm256_fmadd_ps(newDiffY, newDiffY, _mm256_mul_ps(newDiffX, newDiffX));
        __m256 radiusSq = _mm256_mul_ps(destR, destR);

        __m256 reachedMask = _mm256_cmp_ps(newDistSq, radiusSq, _CMP_LT_OQ);
        int maskBits = _mm256_movemask_ps(reachedMask);

        // Update waypoints
        if (maskBits != 0)
        {
            for (int j = 0; j < AGENTS_PER_VECTOR && (i + j) < agentData.count; ++j)
            {
                if (maskBits & (1 << j))
                {
                    int agentIdx = i + j;
                    if (agentData.wpCount[agentIdx] > 0)
                    {
                        int next = agentData.wpIndex[agentIdx] + 1;
                        if (next >= agentData.wpCount[agentIdx])
                            next = 0;
                        agentData.wpIndex[agentIdx] = next;

                        int poolIdx = agentData.wpOffset[agentIdx] + next;
                        agentData.destX[agentIdx] = agentData.wpPoolX[poolIdx];
                        agentData.destY[agentIdx] = agentData.wpPoolY[poolIdx];
                        agentData.destR[agentIdx] = agentData.wpPoolR[poolIdx];
                    }
                }
            }
        }
    }
}

#ifdef USE_CUDA
void Ped::Model::setupCUDA()
{
    cudaStreamCreate(&cudaData.stream);

    size_t floatSize = agentData.paddedCount * sizeof(float);
    size_t intSize = agentData.paddedCount * sizeof(int);
    size_t wpPoolSize = agentData.wpPoolSize * sizeof(float);

    // Allocate device memory
    cudaMalloc(&cudaData.d_x, floatSize);
    cudaMalloc(&cudaData.d_y, floatSize);
    cudaMalloc(&cudaData.d_destX, floatSize);
    cudaMalloc(&cudaData.d_destY, floatSize);
    cudaMalloc(&cudaData.d_destR, floatSize);
    cudaMalloc(&cudaData.d_wpIndex, intSize);
    cudaMalloc(&cudaData.d_wpCount, intSize);
    cudaMalloc(&cudaData.d_wpOffset, intSize);
    cudaMalloc(&cudaData.d_wpPoolX, wpPoolSize);
    cudaMalloc(&cudaData.d_wpPoolY, wpPoolSize);
    cudaMalloc(&cudaData.d_wpPoolR, wpPoolSize);

    // Copy ALL data to GPU once (including initial positions)
    cudaMemcpy(cudaData.d_x, agentData.x, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_y, agentData.y, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_destX, agentData.destX, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_destY, agentData.destY, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_destR, agentData.destR, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpIndex, agentData.wpIndex, intSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpCount, agentData.wpCount, intSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpOffset, agentData.wpOffset, intSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpPoolX, agentData.wpPoolX, wpPoolSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpPoolY, agentData.wpPoolY, wpPoolSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaData.d_wpPoolR, agentData.wpPoolR, wpPoolSize, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

void Ped::Model::tickCUDA()
{
#ifdef USE_CUDA
    // Launch kernel
    cudaKernelfunction(
        cudaData.d_x, cudaData.d_y,
        cudaData.d_destX, cudaData.d_destY, cudaData.d_destR,
        cudaData.d_wpIndex, cudaData.d_wpCount, cudaData.d_wpOffset,
        cudaData.d_wpPoolX, cudaData.d_wpPoolY, cudaData.d_wpPoolR,
        agentData.count, cudaData.stream);

    // ONLY copy positions back (minimal data)
    size_t fSize = agentData.count * sizeof(float); // Use count, not paddedCount
    cudaMemcpyAsync(agentData.x, cudaData.d_x, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.y, cudaData.d_y, fSize, cudaMemcpyDeviceToHost, cudaData.stream);

    cudaStreamSynchronize(cudaData.stream);
#endif
}

void Ped::Model::cleanupCUDA()
{
    static bool cudaCleaned = false;
    if (cudaCleaned)
        return;
    cudaCleaned = true;

    if (implementation == CUDA)
    {
        if (cudaData.stream)
            cudaStreamDestroy(cudaData.stream);
        cudaData.stream = nullptr;
        if (cudaData.d_x)
            cudaFree(cudaData.d_x);
        cudaData.d_x = nullptr;
        if (cudaData.d_y)
            cudaFree(cudaData.d_y);
        cudaData.d_y = nullptr;
        if (cudaData.d_destX)
            cudaFree(cudaData.d_destX);
        cudaData.d_destX = nullptr;
        if (cudaData.d_destY)
            cudaFree(cudaData.d_destY);
        cudaData.d_destY = nullptr;
        if (cudaData.d_destR)
            cudaFree(cudaData.d_destR);
        cudaData.d_destR = nullptr;
        if (cudaData.d_wpIndex)
            cudaFree(cudaData.d_wpIndex);
        cudaData.d_wpIndex = nullptr;
        if (cudaData.d_wpCount)
            cudaFree(cudaData.d_wpCount);
        cudaData.d_wpCount = nullptr;
        if (cudaData.d_wpOffset)
            cudaFree(cudaData.d_wpOffset);
        cudaData.d_wpOffset = nullptr;
        if (cudaData.d_wpPoolX)
            cudaFree(cudaData.d_wpPoolX);
        cudaData.d_wpPoolX = nullptr;
        if (cudaData.d_wpPoolY)
            cudaFree(cudaData.d_wpPoolY);
        cudaData.d_wpPoolY = nullptr;
        if (cudaData.d_wpPoolR)
            cudaFree(cudaData.d_wpPoolR);
        cudaData.d_wpPoolR = nullptr;
    }
}
#endif


// Region setup
void Ped::Model::initRegions()
{
    regions.clear();

    // Start with a 2×2 grid → 4 regions
    int midX = WORLD_WIDTH  / 2;   // 80
    int midY = WORLD_HEIGHT / 2;   // 60

    regions.push_back({0,    midX, 0,    midY});   // top-left
    regions.push_back({midX, WORLD_WIDTH,  0,    midY});   // top-right
    regions.push_back({0,    midX, midY, WORLD_HEIGHT}); // bottom-left
    regions.push_back({midX, WORLD_WIDTH,  midY, WORLD_HEIGHT}); // bottom-right

    rebuildBorderMap();
    assignAgentsToRegions();
}

// Border map logic
// Mark every cell that is within BORDER_WIDTH of any inter-region boundary.
// Calls to this must happen after regions change.

void Ped::Model::rebuildBorderMap()
{
    std::fill(borderCellMap, borderCellMap + WORLD_WIDTH * WORLD_HEIGHT, false);

    for (const auto &r : regions) {
        // Left and right border strips of this region
        for (int y = r.minY; y < r.maxY; ++y) {
            for (int bw = 0; bw < BORDER_WIDTH; ++bw) {
                int xl = r.minX + bw;
                int xr = r.maxX - 1 - bw;
                if (xl < WORLD_WIDTH)  borderCellMap[y * WORLD_WIDTH + xl] = true;
                if (xr >= 0)           borderCellMap[y * WORLD_WIDTH + xr] = true;
            }
        }
        // Top and bottom border strips
        for (int x = r.minX; x < r.maxX; ++x) {
            for (int bw = 0; bw < BORDER_WIDTH; ++bw) {
                int yt = r.minY + bw;
                int yb = r.maxY - 1 - bw;
                if (yt < WORLD_HEIGHT) borderCellMap[yt * WORLD_WIDTH + x] = true;
                if (yb >= 0)           borderCellMap[yb * WORLD_WIDTH + x] = true;
            }
        }
    }
}

void Ped::Model::assignAgentsToRegions()
{
    for (auto &r : regions)
        r.agentIds.clear();

    for (int i = 0; i < agentData.count; ++i) {
        int x = (int)roundf(agentData.x[i]);
        int y = (int)roundf(agentData.y[i]);
        bool placed = false;
        for (auto &r : regions) {
            if (x >= r.minX && x < r.maxX && y >= r.minY && y < r.maxY) {
                r.agentIds.push_back(i);
                placed = true;
                break;
            }
        }
        // Agent out of all regions (shouldn't happen, but be safe)
        if (!placed && !regions.empty())
            regions[0].agentIds.push_back(i);
    }
}

void Ped::Model::tickREGION()
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < agentData.count; ++i) {
        float dx = agentData.destX[i] - agentData.x[i];
        float dy = agentData.destY[i] - agentData.y[i];
        float distSq = dx*dx + dy*dy;

        if (distSq < agentData.destR[i]*agentData.destR[i] && agentData.wpCount[i] > 0) {
            agentData.wpIndex[i] = (agentData.wpIndex[i]+1) % agentData.wpCount[i];
            int pool = agentData.wpOffset[i] + agentData.wpIndex[i];
            agentData.destX[i] = agentData.wpPoolX[pool];
            agentData.destY[i] = agentData.wpPoolY[pool];
            agentData.destR[i] = agentData.wpPoolR[pool];
            dx = agentData.destX[i] - agentData.x[i];
            dy = agentData.destY[i] - agentData.y[i];
            distSq = dx*dx + dy*dy;
        }
        if (distSq > 1e-10f) {
            float inv = 1.0f / sqrtf(distSq);
            agentData.desiredX[i] = agentData.x[i] + dx * inv;
            agentData.desiredY[i] = agentData.y[i] + dy * inv;
        } else {
            agentData.desiredX[i] = agentData.x[i];
            agentData.desiredY[i] = agentData.y[i];
        }
    }

    const int nRegions = (int)regions.size();

#pragma omp parallel
    {
#pragma omp single nowait
        {
            for (int r = 0; r < nRegions; ++r) {
#pragma omp task firstprivate(r)
                {
                    processRegion(r);
                }
            }
        }
    }  // implicit barrier — all tasks complete here

    updateRegions();
}

void Ped::Model::processRegion(int regionIdx)
{
    Region &r = regions[regionIdx];
    for (int agentId : r.agentIds)
        moveInRegion(agentId, r);
}

// Region move
void Ped::Model::moveInRegion(int id, const Region &region)
{
    int currentX = (int)roundf(agentData.x[id]);
    int currentY = (int)roundf(agentData.y[id]);
    int desiredX = (int)roundf(agentData.desiredX[id]);
    int desiredY = (int)roundf(agentData.desiredY[id]);

    if (currentX == desiredX && currentY == desiredY) return;

    int diffX = desiredX - currentX;
    int diffY = desiredY - currentY;

    bool goingRight = (diffX > 0);
    bool goingLeft  = (diffX < 0);
    bool goingDown  = (diffY > 0);
    bool goingUp    = (diffY < 0);

    // Right-hand traffic lane preference (same heuristic as move())
    int lanePreference = 0;
    if      (goingRight) lanePreference = (currentY < 40) ? 0 : -10;
    else if (goingLeft)  lanePreference = (currentY > 80) ? 0 : -10;
    else if (goingDown)  lanePreference = (currentX < 80) ? 0 : -10;
    else if (goingUp)    lanePreference = (currentX > 80) ? 0 : -10;
    bool wrongLane = (lanePreference < 0);

    bool directBlocked = false;
    if (desiredX >= 0 && desiredX < WORLD_WIDTH && desiredY >= 0 && desiredY < WORLD_HEIGHT) {
        directBlocked = (board[desiredY * WORLD_WIDTH + desiredX].load(std::memory_order_relaxed) != -1);
    }

    // Build candidate list (same priority logic as move())
    std::pair<int,int> alts[20];
    int altCount = 0;

    auto addAlt = [&](int x, int y) {
        for (int i = 0; i < altCount; ++i)
            if (alts[i].first == x && alts[i].second == y) return;
        if (altCount < 20) alts[altCount++] = {x, y};
    };

    auto addNormal = [&]() {
        addAlt(desiredX, desiredY);
        if (diffX == 0 || diffY == 0) {
            addAlt(desiredX + diffY, desiredY + diffX);
            addAlt(desiredX - diffY, desiredY - diffX);
        } else {
            addAlt(desiredX, currentY);
            addAlt(currentX, desiredY);
        }
    };

    auto addLane = [&]() {
        if      (goingRight && currentY >= 40) { addAlt(currentX+1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-2); }
        else if (goingLeft  && currentY <= 80) { addAlt(currentX-1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+2); }
        else if (goingDown  && currentX >= 80) { addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-2,currentY+1); }
        else if (goingUp    && currentX <= 80) { addAlt(currentX+1,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX+2,currentY-1); }
    };

    if (/*wrongLane ||*/ directBlocked) { addLane(); addNormal(); }
    else                             { addNormal(); addLane(); }

    int adx = abs(diffX), ady = abs(diffY);
    if      (diffX > 0 && adx >= ady) { addAlt(currentX+1,currentY); addAlt(currentX+1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-1,currentY-1); }
    else if (diffX < 0 && adx >= ady) { addAlt(currentX-1,currentY); addAlt(currentX-1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX+1,currentY+1); }
    else if (diffY < 0 && ady > adx)  { addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX-1,currentY-1); addAlt(currentX-1,currentY); addAlt(currentX,currentY+1); addAlt(currentX+1,currentY+1); }
    else if (diffY > 0 && ady > adx)  { addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX+1,currentY+1); addAlt(currentX+1,currentY); addAlt(currentX,currentY-1); addAlt(currentX-1,currentY-1); }
    else { addAlt(currentX+1,currentY); addAlt(currentX+1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-1); }

    int oldIdx = currentY * WORLD_WIDTH + currentX;

    for (int i = 0; i < altCount; ++i) {
        int px = alts[i].first;
        int py = alts[i].second;
        if (px < 0 || px >= WORLD_WIDTH || py < 0 || py >= WORLD_HEIGHT) continue;

        int targetIdx   = py * WORLD_WIDTH + px;
        bool needAtomic = isBorderCell(px, py) || isBorderCell(currentX, currentY);

        if (needAtomic) {
            int expected = -1;
            if (board[targetIdx].compare_exchange_strong(
                    expected, id,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed))
            {
                // Release the old cell
                if (oldIdx != targetIdx) {
                    int me = id;
                    board[oldIdx].compare_exchange_strong(
                        me, -1,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                }
                agentData.x[id] = (float)px;
                agentData.y[id] = (float)py;
                return;
            }
        } else {
            // Use relaxed loads/stores (plain reads/writes on x86).
            if (board[targetIdx].load(std::memory_order_relaxed) == -1) {
                board[targetIdx].store(id,  std::memory_order_relaxed);
                if (oldIdx != targetIdx)
                    board[oldIdx].store(-1, std::memory_order_relaxed);
                agentData.x[id] = (float)px;
                agentData.y[id] = (float)py;
                return;
            }
        }
    }
    // Agent could not move — stay in place (no starvation risk)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dynamic region management
// ═══════════════════════════════════════════════════════════════════════════════

// ── splitRegion ───────────────────────────────────────────────────────────────
// Splits regions[idx] along its longer axis.  The two halves replace it.
// Returns false if the region is too small to split.

bool Ped::Model::splitRegion(int idx)
{
    // std::cout << "Splitting region " << idx << " with " << regions[idx].agentIds.size() << " agents\n";
    Region &r = regions[idx];
    int w = r.maxX - r.minX;
    int h = r.maxY - r.minY;

    if (w <= MIN_REGION_DIM * 2 && h <= MIN_REGION_DIM * 2) return false;
    if ((int)regions.size() >= MAX_REGIONS)                   return false;

    Region r1 = r, r2 = r;
    if (w >= h) {
        // Split horizontally (along X)
        int mid  = r.minX + w / 2;
        r1.maxX  = mid;
        r2.minX  = mid;
    } else {
        // Split vertically (along Y)
        int mid  = r.minY + h / 2;
        r1.maxY  = mid;
        r2.minY  = mid;
    }
    r1.agentIds.clear();
    r2.agentIds.clear();

    // Replace regions[idx] with r1, append r2
    regions[idx] = r1;
    regions.push_back(r2);
    std::cout << "Split region " << idx << " into two regions with dimensions "
              << "(" << r1.minX << "," << r1.minY << ")-(" << r1.maxX << "," << r1.maxY << ") and "
              << "(" << r2.minX << "," << r2.minY << ")-(" << r2.maxX << "," << r2.maxY << ")\n";
    return true;
}

// ── tryMergeRegions ───────────────────────────────────────────────────────────
// Merges two axis-aligned adjacent regions into one (r1 absorbs r2).
// They must share a complete edge.  Returns false if they are not adjacent.

bool Ped::Model::tryMergeRegions(int r1Idx, int r2Idx)
{
    // std::cout << "Trying to merge regions " << r1Idx << " and " << r2Idx
    //           << " with " << regions[r1Idx].agentIds.size() << " and "
    //           << regions[r2Idx].agentIds.size() << " agents\n";
    Region &r1 = regions[r1Idx];
    Region &r2 = regions[r2Idx];

    // Adjacent along X (r1 right of r2 or r2 right of r1)?
    bool sameY = (r1.minY == r2.minY && r1.maxY == r2.maxY);
    bool sameX = (r1.minX == r2.minX && r1.maxX == r2.maxX);

    if (sameY && r1.maxX == r2.minX) { r1.maxX = r2.maxX; std::cout << "Merged regions " << r1Idx << " and " << r2Idx << std::endl; return true; }
    if (sameY && r2.maxX == r1.minX) { r1.minX = r2.minX; std::cout << "Merged regions " << r1Idx << " and " << r2Idx << std::endl; return true; }
    if (sameX && r1.maxY == r2.minY) { r1.maxY = r2.maxY; std::cout << "Merged regions " << r1Idx << " and " << r2Idx << std::endl; return true; }
    if (sameX && r2.maxY == r1.minY) { r1.minY = r2.minY; std::cout << "Merged regions " << r1Idx << " and " << r2Idx << std::endl; return true; }

    return false;
}

// ── updateRegions ─────────────────────────────────────────────────────────────
// Called at the end of each REGION tick.
//   1. Assign agents to (possibly changed) regions.
//   2. Split overloaded regions.
//   3. Merge underloaded adjacent region pairs.
//   4. Rebuild the border map when the layout changed.

void Ped::Model::updateRegions()
{
    // Always start with a fresh, accurate agent count per region.
    assignAgentsToRegions();

    bool changed = false;

    // ── Split pass ────────────────────────────────────────────────────────────
    // Iterate with index because splitRegion appends to the vector.
    for (int i = 0; i < (int)regions.size(); ++i) {
        if ((int)regions[i].agentIds.size() > SPLIT_THRESHOLD) {
            if (splitRegion(i)) {
                changed = true;
            }
        }
    }

    // ── Re-assign after splits ────────────────────────────────────────────────
    // splitRegion clears agentIds on the two child regions.  Without this call
    // the merge pass below sees those children as empty (0 agents) and
    // immediately merges them back, causing the split-merge thrash visible in
    // the output.
    if (changed) {
        assignAgentsToRegions();
    }

    // ── Merge pass ────────────────────────────────────────────────────────────
    // Merge adjacent pairs that are both sparse AND whose combined population
    // stays below SPLIT_THRESHOLD — otherwise the merged region would be split
    // again on the very next tick.
    bool merged = true;
    while (merged && (int)regions.size() > 4) {  // never drop below 4 regions
        merged = false;
        for (int i = 0; i < (int)regions.size() && !merged; ++i) {
            if ((int)regions[i].agentIds.size() > MERGE_THRESHOLD) continue;
            for (int j = i + 1; j < (int)regions.size() && !merged; ++j) {
                if ((int)regions[j].agentIds.size() > MERGE_THRESHOLD) continue; //TODO Add other logic with agents + agents < splitthreshold

                // Guard: don't create a region that would immediately split again.
                int combined = (int)(regions[i].agentIds.size() + regions[j].agentIds.size());
                if (combined > SPLIT_THRESHOLD) continue;

                if (tryMergeRegions(i, j)) {
                    regions.erase(regions.begin() + j);
                    changed = merged = true;
                }
            }
        }
    }

    if (changed) {
        assignAgentsToRegions();
        rebuildBorderMap();
    }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(int id)
{
    int currentX = (int)roundf(agentData.x[id]);
    int currentY = (int)roundf(agentData.y[id]);
    int desiredX = (int)roundf(agentData.desiredX[id]);
    int desiredY = (int)roundf(agentData.desiredY[id]);

    if (currentX == desiredX && currentY == desiredY) return;

    int diffX = desiredX - currentX;
    int diffY = desiredY - currentY;

    bool goingRight = (diffX > 0);
    bool goingLeft  = (diffX < 0);
    bool goingDown  = (diffY > 0);
    bool goingUp    = (diffY < 0);

    int lanePreference = 0;
    if      (goingRight) lanePreference = (currentY < 40) ? 0 : -10;
    else if (goingLeft)  lanePreference = (currentY > 80) ? 0 : -10;
    else if (goingDown)  lanePreference = (currentX < 80) ? 0 : -10;
    else if (goingUp)    lanePreference = (currentX > 80) ? 0 : -10;
    bool wrongLane = (lanePreference < 0);

    bool directBlocked = false;
    if (desiredX >= 0 && desiredX < WORLD_WIDTH && desiredY >= 0 && desiredY < WORLD_HEIGHT)
        directBlocked = (board[desiredY * WORLD_WIDTH + desiredX].load(std::memory_order_relaxed) != -1);

    std::pair<int,int> alts[20];
    int altCount = 0;

    auto addAlt = [&](int x, int y) {
        for (int i = 0; i < altCount; ++i)
            if (alts[i].first == x && alts[i].second == y) return;
        if (altCount < 20) alts[altCount++] = {x, y};
    };

    auto addNormal = [&]() {
        addAlt(desiredX, desiredY);
        if (diffX == 0 || diffY == 0) { addAlt(desiredX+diffY, desiredY+diffX); addAlt(desiredX-diffY, desiredY-diffX); }
        else                           { addAlt(desiredX, currentY); addAlt(currentX, desiredY); }
    };
    auto addLane = [&]() {
        if      (goingRight && currentY >= 40) { addAlt(currentX+1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-2); }
        else if (goingLeft  && currentY <= 80) { addAlt(currentX-1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+2); }
        else if (goingDown  && currentX >= 80) { addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-2,currentY+1); }
        else if (goingUp    && currentX <= 80) { addAlt(currentX+1,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX+2,currentY-1); }
    };

    if (wrongLane || directBlocked) { addLane(); addNormal(); }
    else                             { addNormal(); addLane(); }

    int adx = abs(diffX), ady = abs(diffY);
    if      (diffX > 0 && adx >= ady) { addAlt(currentX+1,currentY); addAlt(currentX+1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-1,currentY-1); }
    else if (diffX < 0 && adx >= ady) { addAlt(currentX-1,currentY); addAlt(currentX-1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX+1,currentY+1); }
    else if (diffY < 0 && ady > adx)  { addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-1); addAlt(currentX+1,currentY); addAlt(currentX-1,currentY-1); addAlt(currentX-1,currentY); addAlt(currentX,currentY+1); addAlt(currentX+1,currentY+1); }
    else if (diffY > 0 && ady > adx)  { addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX+1,currentY+1); addAlt(currentX+1,currentY); addAlt(currentX,currentY-1); addAlt(currentX-1,currentY-1); }
    else { addAlt(currentX+1,currentY); addAlt(currentX+1,currentY+1); addAlt(currentX,currentY+1); addAlt(currentX-1,currentY+1); addAlt(currentX-1,currentY); addAlt(currentX-1,currentY-1); addAlt(currentX,currentY-1); addAlt(currentX+1,currentY-1); }

    for (int i = 0; i < altCount; ++i) {
        int px = alts[i].first, py = alts[i].second;
        if (px < 0 || px >= WORLD_WIDTH || py < 0 || py >= WORLD_HEIGHT) continue;

        int targetIdx = py * WORLD_WIDTH + px;
        int expected  = -1;
        if (board[targetIdx].compare_exchange_strong(expected, id, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            int oldIdx = currentY * WORLD_WIDTH + currentX;
            if (oldIdx != targetIdx) {
                int me = id;
                board[oldIdx].compare_exchange_strong(me, -1, std::memory_order_release, std::memory_order_relaxed);
            }
            agentData.x[id] = (float)px;
            agentData.y[id] = (float)py;
            return;
        }
    }
}

    
// void Ped::Model::move(Ped::Tagent *agent)
// {
//     int id = agent->getId();
//     // Search for neighboring agents
//     std::set<const Ped::Tagent *> neighbors = getNeighbors((int)agentData.x[id], (int)agentData.y[id], 2);

//     // Retrieve their positions
//     std::vector<std::pair<int, int>> takenPositions;
//     for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
//     {
//         if ((*neighborIt)->getId() != id)
//         {
//             std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
//             takenPositions.push_back(position);
//         }
        
//     }

//     // Compute the three alternative positions that would bring the agent
//     // closer to his desiredPosition, starting with the desiredPosition itself
//     std::vector<std::pair<int, int>> prioritizedAlternatives;
//     std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
//     prioritizedAlternatives.push_back(pDesired);

//     int diffX = pDesired.first - (int)agentData.x[id];
//     int diffY = pDesired.second - (int)agentData.y[id];
//     std::pair<int, int> p1, p2;
//     if (diffX == 0 || diffY == 0)
//     {
//         // Agent wants to walk straight to North, South, West or East
//         p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
//         p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
//     }
//     else
//     {
//         // Agent wants to walk diagonally
//         p1 = std::make_pair(pDesired.first, agentData.y[id]);
//         p2 = std::make_pair(agentData.x[id], pDesired.second);
//     }
//     prioritizedAlternatives.push_back(p1);
//     prioritizedAlternatives.push_back(p2);

//     // Find the first empty alternative position
//     for (std::vector<std::pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
//     {
//         // If the current position is not yet taken by any neighbor
//         if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
//         {
//             // Set the agent's position
//             agentData.x[id] = (float)(*it).first;
//             agentData.y[id] = (float)(*it).second;

//             // agent->setX((*it).first);
//             // agent->setY((*it).second);j
//             break;
//         }
//     }
// }

// void Ped::Model::buildRegions(int minX, int maxX, int minY, int maxY, const std::vector<int>& agentsInRect) 
// {
//     // Thresholds: Max 64 agents per region, minimum size 8x8
//     if (agentsInRect.size() <= 64 || (maxX - minX) <= 8 || (maxY - minY) <= 8) {
//         Region r = {minX, maxX, minY, maxY, agentsInRect};
//         activeRegions.push_back(r);
//         return;
//     }
    
//     // Split into 4 quadrants
//     int midX = minX + (maxX - minX) / 2;
//     int midY = minY + (maxY - minY) / 2;
    
//     std::vector<int> topLeft, topRight, bottomLeft, bottomRight;
    
//     for (int id : agentsInRect) {
//         int ax = (int)agentData.x[id];
//         int ay = (int)agentData.y[id];
        
//         if (ax < midX && ay < midY) topLeft.push_back(id);
//         else if (ax >= midX && ay < midY) topRight.push_back(id);
//         else if (ax < midX && ay >= midY) bottomLeft.push_back(id);
//         else bottomRight.push_back(id);
//     }
    
//     // Recursively build children
//     if (!topLeft.empty()) buildRegions(minX, midX, minY, midY, topLeft);
//     if (!topRight.empty()) buildRegions(midX, maxX, minY, midY, topRight);
//     if (!bottomLeft.empty()) buildRegions(minX, midX, midY, maxY, bottomLeft);
//     if (!bottomRight.empty()) buildRegions(midX, maxX, midY, maxY, bottomRight);
// }

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
std::set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{
    // create the output list
    // ( It would be better to include only the agents close by, but this programmer is lazy.)

    std::set<const Ped::Tagent *> neighbors;
    // float distSq = (float)(dist * dist);

    for(int i = 0; i < agentData.count; i++)
    {
        int curr_positionX = agentData.x[i] - x;
        int curr_positionY = agentData.y[i] - y;

        if (abs(curr_positionX) <= dist && abs(curr_positionY) <= dist)
        {
            neighbors.insert(agents[i]);
        }

        // for(int j = 0; j < agentData.count; j++)
        // {
        //     if(i == j)
        //     {
        //         continue; //ingen anledning att jämföra agent pos med sig själv
        //     }
            
        //     int positionX = agentData.x[j];
        //     int positionY = agentData.y[j];

            

        //     float diff = sqrtf(abs(pow((curr_positionX - positionX), 2.0)) + abs(pow((curr_positionY - positionY), 2.0)));
        //     // float diff = sqrtf(abs(curr_positionX - positionX) * (curr_positionY - positionY));
        //     if(diff <= dist) 
        //     {
        //     // Lägg till i neighbors
        //     // neighbors[i].insert(agents[j]);
            
        //     // neighbors.begin().insert(agents[j]);
        //     *next(neighbors.begin(), i);

            
        //     }
        // }

    }
    
    return neighbors;
    // return std::set<const Ped::Tagent *> (neighbors);
    
    // agentData.destR;
    // agentData.destX;
    // agentData.destY;
    // return std::set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup()
{
    if (isCleaned)
        return;
    isCleaned = true;

    std::cout << "Starting cleanup..." << std::endl;

    // Free aligned memory
    if (agentData.x)
    {
        _mm_free(agentData.x);
        agentData.x = nullptr;
    }
    if (agentData.y)
    {
        _mm_free(agentData.y);
        agentData.y = nullptr;
    }
    if (agentData.destX)
    {
        _mm_free(agentData.destX);
        agentData.destX = nullptr;
    }
    if (agentData.destY)
    {
        _mm_free(agentData.destY);
        agentData.destY = nullptr;
    }
    if (agentData.destR)
    {
        _mm_free(agentData.destR);
        agentData.destR = nullptr;
    }
    if (agentData.wpIndex)
    {
        _mm_free(agentData.wpIndex);
        agentData.wpIndex = nullptr;
    }
    if (agentData.wpCount)
    {
        _mm_free(agentData.wpCount);
        agentData.wpCount = nullptr;
    }
    if (agentData.wpOffset)
    {
        _mm_free(agentData.wpOffset);
        agentData.wpOffset = nullptr;
    }
    if (agentData.wpPoolX)
    {
        _mm_free(agentData.wpPoolX);
        agentData.wpPoolX = nullptr;
    }
    if (agentData.wpPoolY)
    {
        _mm_free(agentData.wpPoolY);
        agentData.wpPoolY = nullptr;
    }
    if (agentData.wpPoolR)
    {
        _mm_free(agentData.wpPoolR);
        agentData.wpPoolR = nullptr;
    }
    if (agentData.desiredX)
    {
        _mm_free(agentData.desiredX);
        agentData.desiredX = nullptr;
    }
    if (agentData.desiredY)
    {
        _mm_free(agentData.desiredY);
        agentData.desiredY = nullptr;
    }

    if (board != nullptr) {
        delete[] board;
        board = nullptr;
    }
    

    std::cout << "Freed all agentData arrays" << std::endl;

#ifdef USE_CUDA
    if (implementation == CUDA)
        cleanupCUDA();
    std::cout << "CUDA cleanup done" << std::endl;
#endif
}

// THE MISSING LINK: The actual implementation of the destructor
Ped::Model::~Model()
{
    std::cout << "Model destructor called" << std::endl;

    // Detach agents so they don't try to access deleted arrays
    for (auto *agent : agents)
    {
        if (agent)
            agent->setId(-1, nullptr);
    }

    cleanup();
    std::cout << "Model destructor finished" << std::endl;
    // DO NOT delete agents or destinations
}