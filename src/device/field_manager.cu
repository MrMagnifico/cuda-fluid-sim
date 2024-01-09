#include "field_manager.cuh"

#include <device/fluid_sim.cuh>
#include <device/gl_interop.cuh>
#include <device/sources.cuh>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>

#include <array>
#include <cuda_gl_interop.h>

FieldManager::FieldManager(const RenderConfig& renderConfig, const uint2 fieldExtents,
                           const GLuint sourcesTex, const GLuint densitiesTex)
    : m_fieldExtents(fieldExtents)
    , m_renderConfig(renderConfig) {
    // Calculate memory sizes and dimensions
    m_paddedfieldExtents    = uint2(fieldExtents.x + 2UL, fieldExtents.y + 2UL);
    m_fieldsSize            = (m_paddedfieldExtents.x)      * (m_paddedfieldExtents.y)      * sizeof(glm::vec3);                                // Account for boundaries
    m_sharedMemSize         = (utils::BLOCK_SIZE.x + 2UL)   * (utils::BLOCK_SIZE.y + 2UL)   * sizeof(glm::vec3) * 2UL;                          // Account for ghost cells and the fact that we store TWO fields (old and new)
    m_gridDims              = dim3((m_paddedfieldExtents.x / utils::BLOCK_SIZE.x) + 1U, (m_paddedfieldExtents.y / utils::BLOCK_SIZE.y) + 1U);   // Grid dimensions needed for workload distribution

    // Allocate and zero initialise memory for fields
    std::array<glm::vec3**, 3UL> fields = { &m_sources, 
                                            &m_densities, &m_densitiesPrev };
    for (auto field : fields) {
        CUDA_ERROR(cudaMalloc(field, m_fieldsSize));
        CUDA_ERROR(cudaMemset(*field, 0, m_fieldsSize));
    }

    // Create OpenGL textures resource handles
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_sourcesResource, sourcesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_densitiesResource, densitiesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

FieldManager::~FieldManager() {
    std::array<glm::vec3*, 3UL> fields = { m_sources, 
                                           m_densities, m_densitiesPrev };
    for (auto field : fields) { CUDA_ERROR(cudaFree(field)); }
}

void FieldManager::copyFieldsToTextures() {
    cudaSurfaceObject_t densitiesSurface    = utils::createSurfaceFromTextureResource(m_densitiesResource);
    cudaSurfaceObject_t sourcesSurface      = utils::createSurfaceFromTextureResource(m_sourcesResource);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densities, densitiesSurface, m_fieldExtents);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_sources, sourcesSurface, m_fieldExtents);
    CUDA_ERROR(cudaDeviceSynchronize()); // Ensure that copying is over before terminating resource handles
    CUDA_ERROR(cudaDestroySurfaceObject(densitiesSurface));
    CUDA_ERROR(cudaDestroySurfaceObject(sourcesSurface));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_densitiesResource));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_sourcesResource));
}

void FieldManager::setSource(uint2 coords, glm::vec3 val) { set_source<<<1, 1>>>(m_sources, coords, val, m_paddedfieldExtents); }

void FieldManager::simulate() {
    add_sources<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densities, m_sources, m_renderConfig.timeStep, m_paddedfieldExtents.x * m_paddedfieldExtents.y);
    std::swap(m_densities, m_densitiesPrev);
    diffuse<<<m_gridDims, utils::BLOCK_SIZE, m_sharedMemSize>>>(m_densitiesPrev, m_densities, m_fieldExtents, m_paddedfieldExtents.x * m_paddedfieldExtents.y,
                                                                m_renderConfig.timeStep, m_renderConfig.diffusionRate, m_renderConfig.diffusionSimSteps);
}
