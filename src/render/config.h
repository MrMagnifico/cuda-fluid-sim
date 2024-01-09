#ifndef _CONFIG_H_
#define _CONFIG_H_


#include <stdint.h>


struct RenderConfig {
    // What to render
    bool renderDensitySources   = false;
    bool renderVelocitySources  = false;
    bool renderDensities        = true;
    bool renderVelocities       = false;

    // Simulation parameters
    float timeStep                = 1e-1f;
    float diffusionRate           = 1e1f;
    uint32_t diffusionSimSteps    = 32U;
    
    // HDR tonemapping and gamma correction
    bool enableHdr  { true };
    float exposure  { 1.0f };
    float gamma     { 2.2f };
};


#endif
