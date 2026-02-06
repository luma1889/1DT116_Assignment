//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cmath>
#include <immintrin.h> 
const unsigned int numThreads = std::thread::hardware_concurrency();

// #define numThreads 11

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

	numAgents = agents.size();
	agentX = (float*)_mm_malloc(numAgents * sizeof(float), 32); // TODO: Check if use this: sizeof(__mm256)
	agentY = (float*)_mm_malloc(numAgents * sizeof(float), 32);
	destX  = (float*)_mm_malloc(numAgents * sizeof(float), 32);
	destY  = (float*)_mm_malloc(numAgents * sizeof(float), 32);
	destR  = (float*)_mm_malloc(numAgents * sizeof(float), 32);
	
	for (size_t i = 0; i < numAgents; ++i)
	{
		agentX[i] = agents[i]->getX();
		agentY[i] = agents[i]->getY();

		Twaypoint* wp = agents[i]->getNextDestination();
		if (wp)
		{
			destX[i] = wp->getx();
			destY[i] = wp->gety();
			destR[i] = wp->getr();
		}
		

	}

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void work(const std::vector<Ped::Tagent *> &agents, size_t start, size_t end)
{
	for (size_t i = start; i < end; ++i)
	{
		agents[i]->computeNextDesiredPosition();
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
	}
}

void Ped::Model::tick()
{
	// EDIT HERE FOR ASSIGNMENT 1
	// enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	switch (implementation)
	{
	case Ped::OMP:
	{
		size_t numAgents = agents.size();

#pragma omp parallel for
		for (size_t i = 0; i < numAgents; ++i)
		{
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
		break;
	}
	case Ped::SEQ:
	{
		size_t numAgents = agents.size();

		for (size_t i = 0; i < numAgents; ++i)
		{
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
		break;
	}
	case Ped::PTHREAD:
	{

		const char *env_t = std::getenv("PTHREAD_NUM_THREADS");
		int numThreads = (env_t != NULL) ? std::stoi(env_t) : 1;

		if (numThreads < 1) numThreads = 1;

		int numAgents = agents.size();
		int chunksize = (numAgents + numThreads - 1) / numThreads;
		std::thread *workers = new std::thread[numThreads];

		for (size_t i = 0; i < numThreads; i++)
		{
			int start = i * chunksize;
			int end = std::min(start + chunksize, numAgents);

			if (start < end)
			{
				workers[i] = std::thread(work, std::ref(agents), start, end);
			}
		}
		for (int i = 0; i < numThreads; i++)
		{
			if (workers[i].joinable())
			{
				workers[i].join();
			}
		}
		delete[] workers;
	}
	case Ped::VECTOR:
	{
		#pragma omp parallel for schedule(static, 8)
		for (int i = 0; i <= numAgents - 8; i += 8)
		{
			bool reachedDest = false;
			__m256 aX = _mm256_load_ps(&agentX[i]);
			__m256 aY = _mm256_load_ps(&agentY[i]);
			__m256 dX = _mm256_load_ps(&destX[i]);
			__m256 dY = _mm256_load_ps(&destY[i]);
			__m256 dR = _mm256_load_ps(&destR[i]);

			__m256 diffX = _mm256_sub_ps(dX, aX);
			__m256 diffY = _mm256_sub_ps(dY, aY);

			// __m256 sq_len = _mm256_add_ps(_mm256_mul_ps(diffX, diffX), _mm256_mul_ps(diffY, diffY));
			__m256 sq_len = _mm256_fmadd_ps(diffX, diffX, _mm256_mul_ps(diffY, diffY));
			// __m256 len = _mm256_sqrt_ps(sq_len);

			// __m256 nextX = _mm256_add_ps(aX, _mm256_div_ps(diffX, len));
			// __m256 nextY = _mm256_add_ps(aY, _mm256_div_ps(diffY, len));
			__m256 invLen = _mm256_rsqrt_ps(sq_len);

			__m256 nextX = _mm256_fmadd_ps(diffX, invLen, aX);
			__m256 nextY = _mm256_fmadd_ps(diffY, invLen, aY);

			__m256 roundedX = _mm256_round_ps(nextX, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			__m256 roundedY = _mm256_round_ps(nextY, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

			_mm256_store_ps (&agentX[i], roundedX);
			_mm256_store_ps (&agentY[i], roundedY);

			for (int j = 0; j < 8; j++)
			{
				agents[i+j]->setX((int)agentX[i+j]);
				agents[i+j]->setY((int)agentY[i+j]);
			}
			

			// Optimized so we only go into scalar if we have reached destination
			__m256 len = _mm256_sqrt_ps(sq_len);
			__m256 mask = _mm256_cmp_ps(len, dR, _CMP_LT_OQ);	// _CMP_LT_OQ = "Less Than, Ordered, Quiet"

			int bitmask = _mm256_movemask_ps(mask);

			if (bitmask != 0) {
				for (int j = 0; j < 8; ++j) {
					if ((bitmask >> j) & 1) {
						Twaypoint* next = agents[i+j]->getNextDestination();
						if (next) {
							destX[i+j] = (float)next->getx();
							destY[i+j] = (float)next->gety();
							destR[i+j] = (float)next->getr();
						}
					}
				}
			}
		}

		// Unoptimized, goes to scalar everythime even though destination is not reached

		// float dists[8];
        // _mm256_storeu_ps(dists, len);
        
        // for (int j = 0; j < 8; ++j) {
        //     // Update the actual object position for GUI
        //     agents[i+j]->setX((int)agentX[i+j]);
        //     agents[i+j]->setY((int)agentY[i+j]);

        //     // If reached destination, get next one and update SoA arrays
        //     if (dists[j] < destR[i+j]) {
        //         Twaypoint* next = agents[i+j]->getNextDestination();
        //         if (next) {
        //             destX[i+j] = (float)next->getx();
        //             destY[i+j] = (float)next->gety();
        //             destR[i+j] = (float)next->getr();
        //         }
        //     }
		
		// For (numAgents % 8)
		int remainder_start = (numAgents / 8) * 8;
		for (int i = remainder_start; i < numAgents; ++i)
		{

			float dx = destX[i] - agentX[i];
			float dy = destY[i] - agentY[i];
			float len = sqrt(dx*dx + dy*dy);
			
			float newX = (int)round(agentX[i] + dx / len);
			float newY = (int)round(agentY[i] + dy / len);

			agents[i]->setX(newX);
    		agents[i]->setY(newY);

			// agents[i]->computeNextDesiredPosition();
			// agents[i]->setX(agents[i]->getDesiredX());
			// agents[i]->setY(agents[i]->getDesiredY());

			agentX[i] = (float)newX;
        	agentY[i] = (float)newY;


			if (len < destR[i]) {
				Twaypoint* next = agents[i]->getNextDestination();
				if (next) {
					destX[i] = (float)next->getx();
					destY[i] = (float)next->gety();
					destR[i] = (float)next->getr();
				}
			}

		}
		break;
	}
	
	default:
		break;
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

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
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
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
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup()
{
	// Nothing to do here right now.
	if (agentX) _mm_free(agentX);
    if (agentY) _mm_free(agentY);
    if (destX)  _mm_free(destX);
    if (destY)  _mm_free(destY);
    if (destR)  _mm_free(destR);
}

Ped::Model::~Model()
{
	cleanup();

	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent)
				  { delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination)
				  { delete destination; });
}
