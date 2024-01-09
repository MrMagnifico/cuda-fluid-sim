#ifndef _FIELD_MANAGER_CUH_
#define _FIELD_MANAGER_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <render/config.h>

class FieldManager {
    public:
        FieldManager(const RenderConfig& renderConfig, const uint2 fieldExtents,
                     const GLuint sourcesTex, const GLuint densitiesTex);
        ~FieldManager();

        void copyFieldsToTextures();
        void setSource(uint2 coords, glm::vec3 val);
        void simulate();

    private:
        const RenderConfig &m_renderConfig;

        uint2 m_fieldExtents;
        uint2 m_paddedfieldExtents;
        size_t m_fieldsSize;
        size_t m_sharedMemSize;
        dim3 m_gridDims;

        // TODO: Expand to other fields
        cudaGraphicsResource_t m_densitiesResource, m_sourcesResource;
        glm::vec3 *m_sources,
                  *m_densities, *m_densitiesPrev;
};


#endif
