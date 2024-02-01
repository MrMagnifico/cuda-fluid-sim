#include "field_manager.cuh"

#include <device/fluid_sim.cuh>
#include <device/fluid_sim.cu>
#include <device/gl_interop.cuh>
#include <device/gl_interop.cu>
#include <device/sources.cuh>
#include <device/sources.cu>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>

#include <array>
#include <cuda_gl_interop.h>

FieldManager::FieldManager(const RenderConfig& renderConfig, const uint2 fieldExtents,
                           const GLuint sourcesDensityTex, const GLuint densitiesTex,
                           const GLuint sourcesVelocityTex, const GLuint velocitiesTex)
    : m_fieldExtents(fieldExtents)
    , m_renderConfig(renderConfig) {
    // Calculate memory sizes and dimensions
    m_paddedfieldExtents    = uint2(fieldExtents.x + 2UL, fieldExtents.y + 2UL);
    m_gridDims              = dim3((m_paddedfieldExtents.x / utils::BLOCK_SIZE.x) + 1U, (m_paddedfieldExtents.y / utils::BLOCK_SIZE.y) + 1U);   // Grid dimensions needed for workload distribution

    // Allocate and zero initialise memory for fields
    std::array<glm::vec4**, utils::FIELDS_PER_TYPE> fieldsDensity   = { &m_densitySources, &m_densities, &m_densitiesPrev };
    std::array<glm::vec2**, utils::FIELDS_PER_TYPE> fieldsVelocity  = { &m_velocitySources, &m_velocities, &m_velocitiesPrev };
    size_t fieldSizeDensity                                         = (m_paddedfieldExtents.x) * (m_paddedfieldExtents.y) * sizeof(glm::vec4); // Account for boundaries
    size_t fieldSizeVelocity                                        = (m_paddedfieldExtents.x) * (m_paddedfieldExtents.y) * sizeof(glm::vec2); // Account for boundaries
    for (size_t fieldIdx = 0UL; fieldIdx < utils::FIELDS_PER_TYPE; fieldIdx++) {
        auto densityField   = fieldsDensity[fieldIdx];
        auto velocityField  = fieldsVelocity[fieldIdx];
        CUDA_ERROR(cudaMalloc(densityField, fieldSizeDensity));
        CUDA_ERROR(cudaMalloc(velocityField, fieldSizeVelocity));
        CUDA_ERROR(cudaMemset(*densityField, 0, fieldSizeDensity));
        CUDA_ERROR(cudaMemset(*velocityField, 0, fieldSizeVelocity));
    }

    // Create texture samplers for desired field(s)
    cudaResourceDesc resourceDesc;
    cudaTextureDesc texDesc;
    setTexObjParams(resourceDesc, texDesc, cudaCreateChannelDesc<float4>(), m_paddedfieldExtents, m_densities);
    CUDA_ERROR(cudaCreateTextureObject(&m_densitiesTex, &resourceDesc, &texDesc, nullptr));
    setTexObjParams(resourceDesc, texDesc, cudaCreateChannelDesc<float4>(), m_paddedfieldExtents, m_densitiesPrev);
    CUDA_ERROR(cudaCreateTextureObject(&m_densitiesPrevTex, &resourceDesc, &texDesc, nullptr));

    // Create OpenGL textures resource handles
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_sourcesDensityResource,   sourcesDensityTex,  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_densitiesResource,        densitiesTex,       GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_sourcesVelocityResource,  sourcesVelocityTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_velocitiesResource,       velocitiesTex,      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

template<typename T>
void FieldManager::setTexObjParams(cudaResourceDesc& resourceDesc, cudaTextureDesc& texDesc,
                                   const cudaChannelFormatDesc& channelDesc, const uint2 fieldExtents,
                                   T* deviceMemory) {
    // Set up the texture description for the specified underlying data type
    memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
    resourceDesc.resType                    = cudaResourceTypePitch2D;
    resourceDesc.res.pitch2D.devPtr         = deviceMemory;
    resourceDesc.res.pitch2D.desc           = channelDesc;
    resourceDesc.res.pitch2D.width          = fieldExtents.x;
    resourceDesc.res.pitch2D.height         = fieldExtents.y;
    resourceDesc.res.pitch2D.pitchInBytes   = fieldExtents.x * sizeof(T); // Assume the width aligns with texture pitch

    // Set up the texture parameters
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0]      = cudaAddressModeClamp;
    texDesc.addressMode[1]      = cudaAddressModeClamp;
    texDesc.filterMode          = cudaFilterModeLinear;
    texDesc.readMode            = cudaReadModeElementType;
    texDesc.normalizedCoords    = true;
}

void FieldManager::copyFieldsToTextures() {
    cudaSurfaceObject_t sourcesDensitySurface   = utils::createSurfaceFromTextureResource(m_sourcesDensityResource);
    cudaSurfaceObject_t densitiesSurface        = utils::createSurfaceFromTextureResource(m_densitiesResource);
    cudaSurfaceObject_t sourcesVelocitySurface  = utils::createSurfaceFromTextureResource(m_sourcesVelocityResource);
    cudaSurfaceObject_t velocitiesSurface       = utils::createSurfaceFromTextureResource(m_velocitiesResource);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densitySources,     sourcesDensitySurface,  m_fieldExtents);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densities,          densitiesSurface,       m_fieldExtents);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocities,         velocitiesSurface,      m_fieldExtents);
    copyFieldToTexture<<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocitySources,    sourcesVelocitySurface, m_fieldExtents);
    CUDA_ERROR(cudaDeviceSynchronize()); // Ensure that copying is over before terminating resource handles
    CUDA_ERROR(cudaDestroySurfaceObject(sourcesDensitySurface));
    CUDA_ERROR(cudaDestroySurfaceObject(densitiesSurface));
    CUDA_ERROR(cudaDestroySurfaceObject(sourcesVelocitySurface));
    CUDA_ERROR(cudaDestroySurfaceObject(velocitiesSurface));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_sourcesDensityResource));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_densitiesResource));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_sourcesVelocityResource));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_velocitiesResource));
}

void FieldManager::setSourceDensity(uint2 coords, glm::vec4 val) { set_source<<<1, 1>>>(m_densitySources, coords, val, m_paddedfieldExtents); }

void FieldManager::setSourceVelocity(uint2 coords, glm::vec2 val) { set_source<<<1, 1>>>(m_velocitySources, coords, val, m_paddedfieldExtents); }

void FieldManager::simulate() {
    velocityStep();
    densityStep();
}

void FieldManager::densityStep() {
    if (m_renderConfig.densityAddSources) { add_sources<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densities, m_densitySources, m_paddedfieldExtents.x * m_paddedfieldExtents.y, m_renderConfig.simulationParams); }
    if (m_renderConfig.densityDiffuse) {
        std::swap(m_densities,      m_densitiesPrev);
        std::swap(m_densitiesTex,   m_densitiesPrevTex);
        size_t sharedMemSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(glm::vec4) * 2UL; // Account for ghost cells and the fact that we store TWO fields (old and new)
        diffuse<<<m_gridDims, utils::BLOCK_SIZE, sharedMemSize>>>(m_densitiesPrev, m_densities, m_fieldExtents, m_paddedfieldExtents.x * m_paddedfieldExtents.y,
                                                                  Conserve, m_renderConfig.simulationParams);
    }
    if (m_renderConfig.densityAdvect) {
        std::swap(m_densities,      m_densitiesPrev);
        std::swap(m_densitiesTex,   m_densitiesPrevTex);
        advect<glm::vec4, float4, glm::vec2><<<m_gridDims, utils::BLOCK_SIZE>>>(m_densitiesPrev, m_densities, m_velocities,
                                                                                m_fieldExtents, m_paddedfieldExtents.x * m_paddedfieldExtents.y,
                                                                                Conserve, m_renderConfig.simulationParams);
    }
}

void FieldManager::velocityStep() {
    if (m_renderConfig.velocityAddSources) { add_sources<<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocities, m_velocitySources, m_paddedfieldExtents.x * m_paddedfieldExtents.y, m_renderConfig.simulationParams); }
    if (m_renderConfig.velocityDiffuse) {
        std::swap(m_velocities, m_velocitiesPrev);
        size_t sharedMemSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(glm::vec2) * 2UL; // Account for ghost cells and the fact that we store TWO fields (old and new)
        diffuse<<<m_gridDims, utils::BLOCK_SIZE, sharedMemSize>>>(m_velocitiesPrev, m_velocities, m_fieldExtents, m_paddedfieldExtents.x * m_paddedfieldExtents.y,
                                                                  Conserve, m_renderConfig.simulationParams);
    }
    if (m_renderConfig.velocityAdvect) {
        std::swap(m_velocities, m_velocitiesPrev);
        advect<glm::vec2, float2, glm::vec2><<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocitiesPrev, m_velocities, m_velocitiesPrev,
                                                                                m_fieldExtents, m_paddedfieldExtents.x * m_paddedfieldExtents.y,
                                                                                Conserve, m_renderConfig.simulationParams);
    }
    if (m_renderConfig.velocityProject) { /* TODO: Add projection step */ }
}