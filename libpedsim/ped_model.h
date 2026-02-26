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
#include <atomic>
#include <numeric>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace Ped {
    class Tagent;

    enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, REGION };

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

            // Temp arrays for movement calculations
            float* desiredX;    // Desired X position (before collision check)
            float* desiredY;    // Desired Y position (before collision check)
            
            int count;          // Actual agent count
            int paddedCount;    // Padded to multiple of 8 for AVX2
        };

        // Region structure for dynamic load balancing
        struct Region {
            int minX, maxX, minY, maxY;
            std::vector<int> agentIds;
        };

        // The Global Lock-Free Board (160x120)
        std::atomic<int>* board;

        static constexpr int BORDER_WIDTH    = 1;   // cells on each region edge treated as border
        static constexpr int SPLIT_THRESHOLD = 128;  // agent count that triggers a split
        static constexpr int MERGE_THRESHOLD = 32;  // agent count below which merging is considered
        static constexpr int MIN_REGION_DIM  = 20;  // minimum width/height of a region (cells)
        static constexpr int MAX_REGIONS     = 48;  // upper cap on dynamic region count

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

        std::vector<int> stuckCounter;  // Track how long each agent has been stuck

        static constexpr int WORLD_WIDTH = 160;
        static constexpr int WORLD_HEIGHT = 120;
        
        // Implementation methods
        void tickSEQ();
        void tickOMP();
        void tickPTHREAD();
        void tickVECTOR();
        void tickCUDA();
        void tickREGION();
        
        // Setup helpers
        void allocateArrays();
        void buildWaypointPool();
        void initializeArrays();

		// Moves an agent towards its next position
		void move(int agentId);

        // Move for regions
        void moveInRegion(int agentId, const Region& region);

        // Rebuilt every time regions change.
        bool borderCellMap[WORLD_WIDTH * WORLD_HEIGHT];
        void rebuildBorderMap();
        inline bool isBorderCell(int x, int y) const {
            if (x < 0 || x >= WORLD_WIDTH || y < 0 || y >= WORLD_HEIGHT) return true;
            return borderCellMap[y * WORLD_WIDTH + x];
        }

        // Region functions
        std::vector<Region> regions;

        // Called once during setup: creates the initial 2×2 = 4 regions.
        void initRegions();

        // O(n) scan that places every agent into its owning region.
        void assignAgentsToRegions();

        // Process all agents inside one region.  Must be called from a single
        // thread per region to guarantee intra-region safety.
        void processRegion(int regionIdx);

        // Dynamic load-balancing: split heavy regions, merge light neighbours.
        // Called at the end of each tickREGION.
        void updateRegions();

        // Split region[idx] along its longer axis at its midpoint.
        // Returns true if the split was performed.
        bool splitRegion(int idx);

        // Try to merge two adjacent, sparse regions.
        // 'r1' and 'r2' must share a complete edge.
        // Returns true if merged (r2 is removed, r1 expanded).
        bool tryMergeRegions(int r1Idx, int r2Idx);


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
            float* d_x = nullptr;
            float* d_y = nullptr; 
            float* d_destX = nullptr;
            float* d_destY = nullptr;
            float* d_destR = nullptr;
            int* d_wpIndex = nullptr;
            int* d_wpCount = nullptr;
            int* d_wpOffset = nullptr;
            float* d_wpPoolX = nullptr;
            float* d_wpPoolY = nullptr;
            float* d_wpPoolR = nullptr;
            cudaStream_t stream = nullptr;
            bool dataValid = false;
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