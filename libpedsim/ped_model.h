//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//

#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include "ped_agent.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace Ped {
    class Tagent;

    enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

    class Model {
    public:
        // Data-Oriented Structure
        struct AgentArrays {
            // Primary arrays (aligned for SIMD)
            float* x;           // Current X positions
            float* y;           // Current Y positions
            float* destX;       // Current destination X
            float* destY;       // Current destination Y
            float* destR;       // Current destination radius
            int* wpIndex;       // Current waypoint index
            int* wpCount;       // Total waypoints per agent
            int* wpOffset;      // Start index in waypoint pool
            
            // Waypoint pool (contiguous memory)
            float* wpPoolX;
            float* wpPoolY;
            float* wpPoolR;
            int wpPoolSize;
            
            int count;          // Actual agent count
            int paddedCount;    // Padded to multiple of 8 for AVX2
        };

        void setup(std::vector<Tagent*> agentsInScenario,
                   std::vector<Twaypoint*> destinationsInScenario,
                   IMPLEMENTATION implementation);
        void tick();
        const std::vector<Tagent*>& getAgents() const { return agents; };
        void cleanup();
        ~Model();

        // Public data for direct access - agents access this via their ID
        AgentArrays agentData;

        // Accessor methods for agents to use
        float getAgentX(int id) const { return agentData.x[id]; }
        float getAgentY(int id) const { return agentData.y[id]; }
        void setAgentX(int id, float value) { agentData.x[id] = value; }
        void setAgentY(int id, float value) { agentData.y[id] = value; }

		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		void placeAgent(const Ped::Tagent *a);

    private:
        IMPLEMENTATION implementation;
        std::vector<Tagent*> agents;
		std::vector<Twaypoint*> destinations;
		bool isCleaned;
        
        // Implementation methods
        void tickSEQ();
        void tickOMP();
        void tickPTHREAD();
        void tickVECTOR();
        void tickCUDA();
        
        // Setup helpers
        void allocateArrays();
        void buildWaypointPool();
        void initializeArrays();

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		std::set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

		#ifdef USE_CUDA
        // CUDA data
        struct CUDAData {
            float* d_x, *d_y, *d_destX, *d_destY, *d_destR;
            int* d_wpIndex, *d_wpCount, *d_wpOffset;
            float* d_wpPoolX, *d_wpPoolY, *d_wpPoolR;
            cudaStream_t stream;
            bool dataValid;  // Track if GPU data needs updating
        } cudaData;
        
        void setupCUDA();
        void cleanupCUDA();
        #endif

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
    };
}
#endif