#ifndef _FIELD_MANAGER_CUH_
#define _FIELD_MANAGER_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

#include <render/config.h>

class FieldManager {
    public:
        FieldManager(const RenderConfig& renderConfig, const uint2 fieldExtents,
                     const GLuint sourcesDensityTex, const GLuint densitiesTex,
                     const GLuint sourcesVelocityTex, const GLuint velocitiesTex);

        void copyFieldsToTextures();
        void setSourceDensity(uint2 coords, glm::vec4 val);
        void setSourceVelocity(uint2 coords, glm::vec2 val);
        void simulate();

    private:
        const RenderConfig &m_renderConfig;

        uint2 m_fieldExtents;
        uint2 m_paddedfieldExtents;
        dim3 m_gridDims;

        cudaTextureObject_t m_densitiesTex, m_densitiesPrevTex, m_velocitiesTex, m_velocitiesPrevTex;
        cudaGraphicsResource_t m_densitiesResource, m_sourcesDensityResource, m_velocitiesResource, m_sourcesVelocityResource;
        glm::vec4 *m_densitySources, *m_densities, *m_densitiesPrev;
        glm::vec2 *m_velocitySources, *m_velocities, *m_velocitiesPrev;

        template<typename T>
        void setTexObjParams(cudaResourceDesc& resourceDesc, cudaTextureDesc& texDesc,
                             const cudaChannelFormatDesc& channelDesc, const uint2 fieldExtents,
                             T* deviceMemory);

        void densityStep();
        void velocityStep();
};


#endif
