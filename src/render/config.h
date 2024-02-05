#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

#include <cstdint>

struct SimulationParams {
    float timeStep              = 1e-2f;
    float diffusionRate         = 1e2f;
    uint32_t diffusionSimSteps  = 16U;
    float advectionMultiplier   = 1e1f;
    uint32_t projectionSimSteps = 16U; 
};

enum BrushEditMode { Densities = 0, DensitySources, Velocities, VelocitySources };

struct BrushParams {
    float scale                 = 0.1f;
    BrushEditMode brushEditMode = Densities;
    glm::vec4 densityDrawColor  = { 1.0f, 0.0f, 0.0f, 1.0f }; 
    glm::vec2 velocityDrawValue = { 1.0f, 1.0f };
    float eraseIntensity        = 0.1f;
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

    // Brush parameters
    BrushParams brushParams;
    
    // HDR tonemapping and gamma correction
    bool enableHdr  = true;
    float exposure  = 1.0f;
    float gamma     = 2.2f;

    // Miscellaneous
    bool enableVsync = false;

    void keyCallback(int key, int scancode, int action, int mods);
};


#endif
