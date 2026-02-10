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
// Declare CUDA kernel function instead of including header
extern "C" void cudaKernelfunction(float* d_x, float* d_y, 
                                   float* d_destX, float* d_destY, float* d_destR,
                                   int* d_wpIndex, const int* d_wpCount, const int* d_wpOffset,
                                   const float* d_wpPoolX, const float* d_wpPoolY, const float* d_wpPoolR,
                                   int numAgents, cudaStream_t stream);
#endif

// Alignment for AVX2 (32 bytes)
#define ALIGNMENT 32
#define AGENTS_PER_VECTOR 8

template<typename T>
T* aligned_alloc(size_t count) {
    return static_cast<T*>(_mm_malloc(count * sizeof(T), ALIGNMENT));
}

void Ped::Model::setup(std::vector<Tagent*> agentsInScenario,
                       std::vector<Twaypoint*> destinationsInScenario,
                       IMPLEMENTATION implementation) {
    this->implementation = implementation;
    this->agents = agentsInScenario;
    this->destinations = destinationsInScenario;
    this->isCleaned = false;
    
    // Setup ID-based architecture
    for (int i = 0; i < (int)agents.size(); ++i) {
        agents[i]->setId(i, this);
    }
    
    // Allocate and initialize arrays
    allocateArrays();
    buildWaypointPool();
    initializeArrays();
    
    // CUDA-specific setup
    if (implementation == CUDA) {
        #ifdef USE_CUDA
        setupCUDA();
        #endif
    }
    
    // setupHeatmapSeq();
}

void Ped::Model::allocateArrays() {
    agentData.count = agents.size();
    agentData.paddedCount = ((agentData.count + AGENTS_PER_VECTOR - 1) / AGENTS_PER_VECTOR) * AGENTS_PER_VECTOR;
    
    // Use cudaMallocHost for ALL arrays - it's pinned and aligned
    cudaMallocHost(&agentData.x, agentData.paddedCount * sizeof(float));
    cudaMallocHost(&agentData.y, agentData.paddedCount * sizeof(float));
    cudaMallocHost(&agentData.destX, agentData.paddedCount * sizeof(float));
    cudaMallocHost(&agentData.destY, agentData.paddedCount * sizeof(float));
    cudaMallocHost(&agentData.destR, agentData.paddedCount * sizeof(float));
    cudaMallocHost(&agentData.wpIndex, agentData.paddedCount * sizeof(int));
    cudaMallocHost(&agentData.wpCount, agentData.paddedCount * sizeof(int));
    cudaMallocHost(&agentData.wpOffset, agentData.paddedCount * sizeof(int));
    
    // Initialize all pointers to nullptr
    agentData.wpPoolX = nullptr;
    agentData.wpPoolY = nullptr;
    agentData.wpPoolR = nullptr;
}

void Ped::Model::buildWaypointPool() {
    // Count total waypoints
    int totalWaypoints = 0;
    for (auto agent : agents) {
        totalWaypoints += std::max(1, (int)agent->getWaypoints().size());
    }
    
    agentData.wpPoolSize = totalWaypoints;
    
    // Use cudaMallocHost for consistency
    cudaMallocHost(&agentData.wpPoolX, totalWaypoints * sizeof(float));
    cudaMallocHost(&agentData.wpPoolY, totalWaypoints * sizeof(float));
    cudaMallocHost(&agentData.wpPoolR, totalWaypoints * sizeof(float));
    
    // Fill waypoint pool
    int offset = 0;
    for (int i = 0; i < agentData.count; ++i) {
        agentData.wpOffset[i] = offset;
        agentData.wpIndex[i] = 0;
        
        const auto& waypoints = agents[i]->getWaypoints();
        agentData.wpCount[i] = std::max(1, (int)waypoints.size());
        
        if (waypoints.empty()) {
            // No waypoints - stay in place
            agentData.wpPoolX[offset] = (float)agents[i]->getInitX();
            agentData.wpPoolY[offset] = (float)agents[i]->getInitY();
            agentData.wpPoolR[offset] = 1.0f;
            offset++;
        } else {
            for (auto wp : waypoints) {
                agentData.wpPoolX[offset] = (float)wp->getx();
                agentData.wpPoolY[offset] = (float)wp->gety();
                agentData.wpPoolR[offset] = (float)wp->getr();
                offset++;
            }
        }
    }
}

void Ped::Model::initializeArrays() {
    // Initialize arrays with agent data
    for (int i = 0; i < agentData.count; ++i) {
        agentData.x[i] = (float)agents[i]->getInitX();
        agentData.y[i] = (float)agents[i]->getInitY();
        
        // Set initial destination
        int wpIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
        agentData.destX[i] = agentData.wpPoolX[wpIdx];
        agentData.destY[i] = agentData.wpPoolY[wpIdx];
        agentData.destR[i] = agentData.wpPoolR[wpIdx];
    }
    
    // Pad remaining elements
    for (int i = agentData.count; i < agentData.paddedCount; ++i) {
        agentData.x[i] = 0.0f;
        agentData.y[i] = 0.0f;
        agentData.destX[i] = 0.0f;
        agentData.destY[i] = 0.0f;
        agentData.destR[i] = 1.0f;
        agentData.wpIndex[i] = 0;
        agentData.wpCount[i] = 1;
        agentData.wpOffset[i] = 0;
    }
}

void Ped::Model::tick() {
    switch (implementation) {
        case SEQ: tickSEQ(); break;
        case OMP: tickOMP(); break;
        case PTHREAD: tickPTHREAD(); break;
        case VECTOR: tickVECTOR(); break;
        case CUDA: tickCUDA(); break;
    }
    // updateHeatmapSeq();
}

// Optimized sequential implementation
void Ped::Model::tickSEQ() {
    for (int i = 0; i < agentData.count; ++i) {
        float dx = agentData.destX[i] - agentData.x[i];
        float dy = agentData.destY[i] - agentData.y[i];
        float distSq = dx * dx + dy * dy;
        
        if (distSq > 1e-10f) {
            float invDist = 1.0f / sqrtf(distSq);
            agentData.x[i] += dx * invDist;
            agentData.y[i] += dy * invDist;
        }
        
        dx = agentData.destX[i] - agentData.x[i];
        dy = agentData.destY[i] - agentData.y[i];
        if ((dx * dx + dy * dy) < (agentData.destR[i] * agentData.destR[i])) {
            if (agentData.wpCount[i] > 0) {
                agentData.wpIndex[i] = (agentData.wpIndex[i] + 1) % agentData.wpCount[i];
                int poolIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
                agentData.destX[i] = agentData.wpPoolX[poolIdx];
                agentData.destY[i] = agentData.wpPoolY[poolIdx];
                agentData.destR[i] = agentData.wpPoolR[poolIdx];
            }
        }
    }
}

// Optimized OpenMP implementation
void Ped::Model::tickOMP() {
    #pragma omp parallel for
    for (int i = 0; i < agentData.count; ++i) {
        float dx = agentData.destX[i] - agentData.x[i];
        float dy = agentData.destY[i] - agentData.y[i];
        float distSq = dx * dx + dy * dy;
        
        if (distSq > 1e-10f) {
            float invDist = 1.0f / sqrtf(distSq);
            agentData.x[i] += dx * invDist;
            agentData.y[i] += dy * invDist;
        }
        
        dx = agentData.destX[i] - agentData.x[i];
        dy = agentData.destY[i] - agentData.y[i];
        if ((dx * dx + dy * dy) < (agentData.destR[i] * agentData.destR[i])) {
            if (agentData.wpCount[i] > 0) {
                agentData.wpIndex[i] = (agentData.wpIndex[i] + 1) % agentData.wpCount[i];
                int poolIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
                agentData.destX[i] = agentData.wpPoolX[poolIdx];
                agentData.destY[i] = agentData.wpPoolY[poolIdx];
                agentData.destR[i] = agentData.wpPoolR[poolIdx];
            }
        }
    }
}

// PThread implementation
void Ped::Model::tickPTHREAD() {
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = (agentData.count + num_threads - 1) / num_threads;
    
    auto worker = [this](int start, int end) {
        for (int i = start; i < end; ++i) {
            float dx = agentData.destX[i] - agentData.x[i];
            float dy = agentData.destY[i] - agentData.y[i];
            float distSq = dx * dx + dy * dy;
            
            if (distSq > 1e-10f) {
                float invDist = 1.0f / sqrtf(distSq);
                agentData.x[i] += dx * invDist;
                agentData.y[i] += dy * invDist;
            }
            
            dx = agentData.destX[i] - agentData.x[i];
            dy = agentData.destY[i] - agentData.y[i];
            if ((dx * dx + dy * dy) < (agentData.destR[i] * agentData.destR[i])) {
                if (agentData.wpCount[i] > 0) {
                    agentData.wpIndex[i] = (agentData.wpIndex[i] + 1) % agentData.wpCount[i];
                    int poolIdx = agentData.wpOffset[i] + agentData.wpIndex[i];
                    agentData.destX[i] = agentData.wpPoolX[poolIdx];
                    agentData.destY[i] = agentData.wpPoolY[poolIdx];
                    agentData.destR[i] = agentData.wpPoolR[poolIdx];
                }
            }
        }
    };
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, agentData.count);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    
    for (auto& t : threads) t.join();
}

// OPTIMIZED VECTOR IMPLEMENTATION (Fully Vectorized)
void Ped::Model::tickVECTOR() {
    // Process 8 agents at a time using AVX2
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < agentData.paddedCount; i += AGENTS_PER_VECTOR) {
        // Load positions and destinations
        __m256 posX = _mm256_load_ps(&agentData.x[i]);
        __m256 posY = _mm256_load_ps(&agentData.y[i]);
        __m256 destX = _mm256_load_ps(&agentData.destX[i]);
        __m256 destY = _mm256_load_ps(&agentData.destY[i]);
        __m256 destR = _mm256_load_ps(&agentData.destR[i]);
        
        // Calculate direction vectors
        __m256 diffX = _mm256_sub_ps(destX, posX);
        __m256 diffY = _mm256_sub_ps(destY, posY);
        
        // Calculate squared distance
        __m256 dx2 = _mm256_mul_ps(diffX, diffX);
        __m256 dy2 = _mm256_mul_ps(diffY, diffY);
        __m256 distSq = _mm256_add_ps(dx2, dy2);
        
        // Calculate normalized movement (avoid division by zero)
        __m256 mask = _mm256_cmp_ps(distSq, _mm256_set1_ps(1e-10f), _CMP_GT_OQ);
        __m256 invDist = _mm256_rsqrt_ps(distSq);
        invDist = _mm256_and_ps(invDist, mask);
        
        // Update positions
        __m256 stepX = _mm256_mul_ps(diffX, invDist);
        __m256 stepY = _mm256_mul_ps(diffY, invDist);
        posX = _mm256_add_ps(posX, stepX);
        posY = _mm256_add_ps(posY, stepY);
        
        // Store updated positions
        _mm256_store_ps(&agentData.x[i], posX);
        _mm256_store_ps(&agentData.y[i], posY);
        
        // Check if destinations reached (vectorized)
        __m256 newDiffX = _mm256_sub_ps(destX, posX);
        __m256 newDiffY = _mm256_sub_ps(destY, posY);
        __m256 newDx2 = _mm256_mul_ps(newDiffX, newDiffX);
        __m256 newDy2 = _mm256_mul_ps(newDiffY, newDiffY);
        __m256 newDistSq = _mm256_add_ps(newDx2, newDy2);
        __m256 radiusSq = _mm256_mul_ps(destR, destR);
        
        __m256 reachedMask = _mm256_cmp_ps(newDistSq, radiusSq, _CMP_LT_OQ);
        int maskBits = _mm256_movemask_ps(reachedMask);
        
        // Update waypoints for agents that reached destination
        if (maskBits != 0) {
            for (int j = 0; j < AGENTS_PER_VECTOR && (i + j) < agentData.count; ++j) {
                if (maskBits & (1 << j)) {
                    int agentIdx = i + j;
                    if (agentData.wpCount[agentIdx] > 0) {
                        agentData.wpIndex[agentIdx] = (agentData.wpIndex[agentIdx] + 1) % agentData.wpCount[agentIdx];
                        int poolIdx = agentData.wpOffset[agentIdx] + agentData.wpIndex[agentIdx];
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
void Ped::Model::setupCUDA() {
    cudaStreamCreate(&cudaData.stream);
    cudaData.dataValid = false;
    
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
    
    // Copy static data to GPU
    cudaMemcpyAsync(cudaData.d_wpPoolX, agentData.wpPoolX, wpPoolSize, 
                    cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_wpPoolY, agentData.wpPoolY, wpPoolSize, 
                    cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_wpPoolR, agentData.wpPoolR, wpPoolSize, 
                    cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_wpCount, agentData.wpCount, intSize, 
                    cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_wpOffset, agentData.wpOffset, intSize, 
                    cudaMemcpyHostToDevice, cudaData.stream);
    
    // cudaStreamSynchronize(cudaData.stream);
}

void Ped::Model::tickCUDA() {
    #ifdef USE_CUDA
    size_t fSize = agentData.paddedCount * sizeof(float);
    size_t iSize = agentData.paddedCount * sizeof(int);
    
    // 1. Copy dynamic data TO GPU (Slide 42: HostToDevice)
    // We only copy what is needed for the math
    cudaMemcpyAsync(cudaData.d_x, agentData.x, fSize, cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_y, agentData.y, fSize, cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_destX, agentData.destX, fSize, cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_destY, agentData.destY, fSize, cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_destR, agentData.destR, fSize, cudaMemcpyHostToDevice, cudaData.stream);
    cudaMemcpyAsync(cudaData.d_wpIndex, agentData.wpIndex, iSize, cudaMemcpyHostToDevice, cudaData.stream);

    // 2. Launch Kernel (Slide 42)
    cudaKernelfunction(
        cudaData.d_x, cudaData.d_y, cudaData.d_destX, cudaData.d_destY, cudaData.d_destR,
        cudaData.d_wpIndex, cudaData.d_wpCount, cudaData.d_wpOffset,
        cudaData.d_wpPoolX, cudaData.d_wpPoolY, cudaData.d_wpPoolR,
        agentData.count, cudaData.stream);
    
    // 3. Copy results BACK to CPU (Slide 42: DeviceToHost)
    // We need the new X/Y for the GUI and the new wpIndex to know where agents are
    cudaMemcpyAsync(agentData.x, cudaData.d_x, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.y, cudaData.d_y, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.destX, cudaData.d_destX, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.destY, cudaData.d_destY, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.destR, cudaData.d_destR, fSize, cudaMemcpyDeviceToHost, cudaData.stream);
    cudaMemcpyAsync(agentData.wpIndex, cudaData.d_wpIndex, iSize, cudaMemcpyDeviceToHost, cudaData.stream);
    
    // 4. Synchronize the stream before the tick ends
    cudaStreamSynchronize(cudaData.stream);
    #endif
}

void Ped::Model::cleanupCUDA() {
    static bool cudaCleaned = false;
    if (cudaCleaned) return;
    cudaCleaned = true;
    
    if (implementation == CUDA) {
        if(cudaData.stream) cudaStreamDestroy(cudaData.stream); cudaData.stream = nullptr;
        if(cudaData.d_x) cudaFree(cudaData.d_x); cudaData.d_x = nullptr;
        if(cudaData.d_y) cudaFree(cudaData.d_y); cudaData.d_y = nullptr;
        if(cudaData.d_destX) cudaFree(cudaData.d_destX); cudaData.d_destX = nullptr;
        if(cudaData.d_destY) cudaFree(cudaData.d_destY); cudaData.d_destY = nullptr;
        if(cudaData.d_destR) cudaFree(cudaData.d_destR); cudaData.d_destR = nullptr;
        if(cudaData.d_wpIndex) cudaFree(cudaData.d_wpIndex); cudaData.d_wpIndex = nullptr;
        if(cudaData.d_wpCount) cudaFree(cudaData.d_wpCount); cudaData.d_wpCount = nullptr;
        if(cudaData.d_wpOffset) cudaFree(cudaData.d_wpOffset); cudaData.d_wpOffset = nullptr;
        if(cudaData.d_wpPoolX) cudaFree(cudaData.d_wpPoolX); cudaData.d_wpPoolX = nullptr;
        if(cudaData.d_wpPoolY) cudaFree(cudaData.d_wpPoolY); cudaData.d_wpPoolY = nullptr;
        if(cudaData.d_wpPoolR) cudaFree(cudaData.d_wpPoolR); cudaData.d_wpPoolR = nullptr;
    }
}
#endif



////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	std::set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
	{
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else
	{
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<std::pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
	{
		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
		{
			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);
			break;
		}
	}
}

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
	return std::set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
    if (isCleaned) return;
    isCleaned = true;
    
    std::cout << "Starting cleanup..." << std::endl;
    
    // Free pinned memory allocated with cudaMallocHost
    if (agentData.x)        { cudaFreeHost(agentData.x);        agentData.x = nullptr; }
    if (agentData.y)        { cudaFreeHost(agentData.y);        agentData.y = nullptr; }
    if (agentData.destX)    { cudaFreeHost(agentData.destX);    agentData.destX = nullptr; }
    if (agentData.destY)    { cudaFreeHost(agentData.destY);    agentData.destY = nullptr; }
    if (agentData.destR)    { cudaFreeHost(agentData.destR);    agentData.destR = nullptr; }
    if (agentData.wpIndex)  { cudaFreeHost(agentData.wpIndex);  agentData.wpIndex = nullptr; }
    if (agentData.wpCount)  { cudaFreeHost(agentData.wpCount);  agentData.wpCount = nullptr; }
    if (agentData.wpOffset) { cudaFreeHost(agentData.wpOffset); agentData.wpOffset = nullptr; }
    if (agentData.wpPoolX)  { cudaFreeHost(agentData.wpPoolX);  agentData.wpPoolX = nullptr; }
    if (agentData.wpPoolY)  { cudaFreeHost(agentData.wpPoolY);  agentData.wpPoolY = nullptr; }
    if (agentData.wpPoolR)  { cudaFreeHost(agentData.wpPoolR);  agentData.wpPoolR = nullptr; }
    
    std::cout << "Freed all agentData arrays" << std::endl;
    
    #ifdef USE_CUDA
    if (implementation == CUDA) cleanupCUDA();
    std::cout << "CUDA cleanup done" << std::endl;
    #endif
}

// THE MISSING LINK: The actual implementation of the destructor
Ped::Model::~Model() {
    std::cout << "Model destructor called" << std::endl;
    
    // Detach agents so they don't try to access deleted arrays
    for(auto* agent : agents) {
        if(agent) agent->setId(-1, nullptr); 
    }

    cleanup();
    std::cout << "Model destructor finished" << std::endl;
    // DO NOT delete agents or destinations
}

