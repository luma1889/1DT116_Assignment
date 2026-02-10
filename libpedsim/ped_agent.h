//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <cmath>

namespace Ped {
    class Model;
    class Twaypoint;

    class Tagent {
    public:
        // Data-Oriented Design: Agent is just a handle to arrays in Model
		Tagent(int posX, int posY);
        Tagent(double posX, double posY);
        // Tagent(int posX, int posY) : id(-1), model(nullptr), x(posX), y(posY) {}
        // Tagent(double posX, double posY) : 
        //     id(-1), model(nullptr), x((int)round(posX)), y((int)round(posY)) {}
        ~Tagent() {
        // Don't access model data during destruction
			model = nullptr;
			id = -1;
		}
        // Getters/Setters - direct array access
        int getX() const;
        int getY() const;
        void setX(int newX);
        void setY(int newY);
        
        // For compatibility
        int getDesiredX() const { return getX(); }
        int getDesiredY() const { return getY(); }
        void computeNextDesiredPosition() {}  // Handled by Model
        
        // Waypoint management
        void addWaypoint(Twaypoint* wp);
        Twaypoint* getNextDestination();
        const std::deque<Twaypoint*>& getWaypoints() const { return tmp_waypoints; }
        
        // ID-based architecture
        int getId() const { return id; }
        void setId(int newId, Model* m) { id = newId; model = m; }
        
        // Temporary storage for setup
        int getInitX() const { return init_x; }
        int getInitY() const { return init_y; }
        
    private:
        int id;             // Index in Model's arrays
        Model* model;       // Pointer to source of truth
        int init_x, init_y; // Initial positions
        std::deque<Twaypoint*> tmp_waypoints;  // Only used during setup
        
        void init(int posX, int posY);
    };
}
#endif