#ifndef _FIELD_MANAGER_CUH_
#define _FIELD_MANAGER_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <render/config.h>

class FieldManager {
    public:
        FieldManager(const uint2 fieldExtents, const RenderConfig& renderConfig);
        ~FieldManager();

        void copyFieldsToTexture(cudaSurfaceObject_t sourcesSurface, cudaSurfaceObject_t densitiesSurface);
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
        glm::vec3 *m_sources,
                  *m_densities, *m_densitiesPrev;
};


#endif
