#ifndef _FIELD_MANAGER_CUH_
#define _FIELD_MANAGER_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

#include <framework/window.h>
#include <render/config.h>

struct BoundingBox {
    uint2 topLeft, bottomRight;
};

class FieldManager {
    public:
        FieldManager(const RenderConfig& renderConfig, const Window &window,
                     const uint2 fieldExtents,
                     const GLuint sourcesDensityTex, const GLuint densitiesTex,
                     const GLuint sourcesVelocityTex, const GLuint velocitiesTex);

        void copyFieldsToTextures();
        void simulate();

        void mouseButtonCallback(int button, int action, int mods);
        void mouseMoveCallback(glm::vec2 cursorPos);
        void setSourceDensity(uint2 coords, glm::vec4 val);
        void setSourceVelocity(uint2 coords, glm::vec2 val);

    private:
        const RenderConfig &m_renderConfig;
        const Window &m_window;

        uint2 m_fieldExtents;
        uint2 m_paddedfieldExtents;
        dim3 m_gridDims;

        cudaGraphicsResource_t m_densitiesResource, m_sourcesDensityResource, m_velocitiesResource, m_sourcesVelocityResource;
        glm::vec4 *m_densitySources, *m_densities, *m_densitiesPrev;
        glm::vec2 *m_velocitySources, *m_velocities, *m_velocitiesPrev;

        void densityStep();
        void velocityStep();

        BoundingBox brushBoundingBox();
        dim3 brushGridDims(BoundingBox boundingBox);
        void applyBrushAdditive();
        void applyBrushErase();
};


#endif
