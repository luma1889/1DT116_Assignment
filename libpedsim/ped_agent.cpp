//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_model.h"
#include "ped_waypoint.h"
#include <math.h>

Ped::Tagent::Tagent(int posX, int posY) {
    init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
    init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
    id = -1;
    model = nullptr;
    init_x = posX;
    init_y = posY;
}

int Ped::Tagent::getX() const {
    if (model && id >= 0) {
        // Direct array access - no sync needed!
        return (int)model->agentData.x[id];
    }
    return init_x;  // Fallback during setup
}

int Ped::Tagent::getY() const {
    if (model && id >= 0) {
        return (int)model->agentData.y[id];
    }
    return init_y;
}

void Ped::Tagent::setX(int newX) {
    if (model && id >= 0) {
        model->agentData.x[id] = (float)newX;
    } else {
        init_x = newX;
    }
}

void Ped::Tagent::setY(int newY) {
    if (model && id >= 0) {
        model->agentData.y[id] = (float)newY;
    } else {
        init_y = newY;
    }
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
    tmp_waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
    if (!tmp_waypoints.empty()) {
        return tmp_waypoints.front();
    }
    return nullptr;
}