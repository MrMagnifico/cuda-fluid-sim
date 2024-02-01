#ifndef _CONFIG_H_
#define _CONFIG_H_


#include <stdint.h>

struct SimulationParams {
    float timeStep                = 1e-1f;
    float diffusionRate           = 2.0f;
    uint32_t diffusionSimSteps    = 32U;
    float advectionMultiplier     = 1e-3f;
};

struct RenderConfig {
    // What to render
    bool renderDensitySources   = false;
    bool renderVelocitySources  = false;
    bool renderDensities        = true;
    bool renderVelocities       = false;

    // Simulation step toggles
    bool densityAddSources  = false;
    bool densityDiffuse     = false;
    bool densityAdvect      = false;
    bool velocityAddSources = false;
    bool velocityDiffuse    = false;
    bool velocityAdvect     = false;
    bool velocityProject    = false;

    // Simulation parameters
    SimulationParams simulationParams;
    
    // HDR tonemapping and gamma correction
    bool enableHdr  = true;
    float exposure  = 1.0f;
    float gamma     = 2.2f;
};


#endif
