#include "field_manager.cuh"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <GLFW/glfw3.h>
DISABLE_WARNINGS_POP()

#include <device/fluid_sim.cuh>
#include <device/fluid_sim.cu>
#include <device/gl_interop.cuh>
#include <device/gl_interop.cu>
#include <device/field_edit.cuh>
#include <device/field_edit.cu>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>

#include <array>
#include <cuda_gl_interop.h>

FieldManager::FieldManager(const RenderConfig& renderConfig, const Window &window,
                           const uint2 fieldExtents,
                           const GLuint sourcesDensityTex, const GLuint densitiesTex,
                           const GLuint sourcesVelocityTex, const GLuint velocitiesTex)
    : m_fieldExtents(fieldExtents)
    , m_renderConfig(renderConfig)
    , m_window(window) {
    // Calculate memory sizes and dimensions
    m_paddedfieldExtents    = uint2(fieldExtents.x + 2UL, fieldExtents.y + 2UL);
    m_gridDims              = dim3((m_paddedfieldExtents.x / utils::BLOCK_SIZE.x) + 1U, (m_paddedfieldExtents.y / utils::BLOCK_SIZE.y) + 1U);   // Grid dimensions needed for workload distribution

    // Allocate and zero initialise memory for fields
    std::array<glm::vec3**, utils::FIELDS_PER_TYPE> fieldsDensity   = { &m_densitySources, &m_densities, &m_densitiesPrev };
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

    // Create OpenGL textures resource handles
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_sourcesDensityResource,   sourcesDensityTex,  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_densitiesResource,        densitiesTex,       GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_sourcesVelocityResource,  sourcesVelocityTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_velocitiesResource,       velocitiesTex,      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
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

void FieldManager::mouseButtonCallback(int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)   { applyBrushAdditive(); }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)  { applyBrushErase(); }
}

void FieldManager::mouseMoveCallback(glm::vec2 cursorPos) {
    if (m_window.isMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT))  { applyBrushAdditive(); }
    if (m_window.isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT)) { applyBrushErase(); }
}

void FieldManager::applyBrushAdditive() {
    BoundingBox brushBB = brushBoundingBox();
    dim3 brushGrid      = brushGridDims(brushBB);
    switch (m_renderConfig.brushParams.brushEditMode) {
        case Densities: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_densities, m_renderConfig.brushParams.densityDrawColor,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Add, false);
        } break;
        case DensitySources: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_densitySources, m_renderConfig.brushParams.densityDrawColor,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Add, false);
        } break;
        case Velocities: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_velocities, m_renderConfig.brushParams.velocityDrawValue,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Add, false);
        } break;
        case VelocitySources: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_velocitySources, m_renderConfig.brushParams.velocityDrawValue,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Add, false);
        } break;
    }
}

void FieldManager::applyBrushErase() {
    BoundingBox brushBB = brushBoundingBox();
    dim3 brushGrid      = brushGridDims(brushBB);
    switch (m_renderConfig.brushParams.brushEditMode) {
        case Densities: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_densities, glm::vec3(m_renderConfig.brushParams.eraseIntensity),
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Remove, true);
        } break;
        case DensitySources: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_densitySources, glm::vec3(m_renderConfig.brushParams.eraseIntensity),
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Remove, true);
        } break;
        case Velocities: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_velocities, m_renderConfig.brushParams.velocityDrawValue,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Remove, false);
        } break;
        case VelocitySources: {
            update_field<<<brushGrid, utils::BLOCK_SIZE>>>(m_velocitySources, m_renderConfig.brushParams.velocityDrawValue,
                                                           m_fieldExtents, brushBB.topLeft, brushBB.bottomRight,
                                                           Remove, false);
        } break;
    }
}

void FieldManager::setSourceDensity(uint2 coords, glm::vec3 val) { set_source<<<1, 1>>>(m_densitySources, coords, val, m_paddedfieldExtents); }

void FieldManager::setSourceVelocity(uint2 coords, glm::vec2 val) { set_source<<<1, 1>>>(m_velocitySources, coords, val, m_paddedfieldExtents); }

void FieldManager::simulate() {
    velocityStep();
    densityStep();
}

void FieldManager::densityStep() {
    if (m_renderConfig.densityAddSources) { add_sources<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densities, m_densitySources, m_fieldExtents, m_renderConfig.simulationParams); }
    if (m_renderConfig.densityDiffuse) {
        std::swap(m_densities,      m_densitiesPrev);
        size_t sharedMemSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(glm::vec4) * 2UL; // Account for ghost cells and the fact that we store TWO fields (old and new)
        diffuse<<<m_gridDims, utils::BLOCK_SIZE, sharedMemSize>>>(m_densitiesPrev, m_densities, m_fieldExtents, Conserve, m_renderConfig.simulationParams);
    }
    if (m_renderConfig.densityAdvect) {
        std::swap(m_densities,      m_densitiesPrev);
        advect<<<m_gridDims, utils::BLOCK_SIZE>>>(m_densitiesPrev, m_densities, m_velocities, m_fieldExtents, Conserve, m_renderConfig.simulationParams);
    }
}

void FieldManager::velocityStep() {
    if (m_renderConfig.velocityAddSources) { add_sources<<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocities, m_velocitySources, m_fieldExtents, m_renderConfig.simulationParams); }
    if (m_renderConfig.velocityDiffuse) {
        std::swap(m_velocities, m_velocitiesPrev);
        size_t sharedMemDiffuseSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(glm::vec2) * 2UL; // Account for ghost cells and the fact that we store TWO fields (old and new)
        diffuse<<<m_gridDims, utils::BLOCK_SIZE, sharedMemDiffuseSize>>>(m_velocitiesPrev, m_velocities, m_fieldExtents, Reverse, m_renderConfig.simulationParams);
        if (m_renderConfig.velocityProject) {
            size_t sharedMemProjectSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(float) * 2UL; // Account for ghost cells and the fact that we store TWO fields (derivative and projection)
            project<<<m_gridDims, utils::BLOCK_SIZE, sharedMemProjectSize>>>(m_velocities, m_fieldExtents, m_renderConfig.simulationParams);
        }
    }
    if (m_renderConfig.velocityAdvect) {
        std::swap(m_velocities, m_velocitiesPrev);
        advect<<<m_gridDims, utils::BLOCK_SIZE>>>(m_velocitiesPrev, m_velocities, m_velocitiesPrev, m_fieldExtents, Reverse, m_renderConfig.simulationParams);
        if (m_renderConfig.velocityProject) {
            size_t sharedMemProjectSize = (utils::BLOCK_SIZE.x + 2UL) * (utils::BLOCK_SIZE.y + 2UL) * sizeof(float) * 2UL; // Account for ghost cells and the fact that we store TWO fields (derivative and projection)
            project<<<m_gridDims, utils::BLOCK_SIZE, sharedMemProjectSize>>>(m_velocities, m_fieldExtents, m_renderConfig.simulationParams);
        }
    }
}

BoundingBox FieldManager::brushBoundingBox() {
    // Compute unbounded bounding box based on current cursor position
    glm::vec2 cursorPosition    = m_window.getNormalizedCursorPos() * glm::vec2(m_fieldExtents.x, m_fieldExtents.y);
    cursorPosition.y            = m_fieldExtents.y - cursorPosition.y; // Place origin at top-left
    float halfScale             = 0.5f * m_renderConfig.brushParams.scale;
    glm::vec2 topLeft(cursorPosition.x - halfScale * m_fieldExtents.x,
                      cursorPosition.y - halfScale * m_fieldExtents.y * m_window.getAspectRatio());
    glm::vec2 bottomRight(cursorPosition.x + halfScale * m_fieldExtents.x,
                          cursorPosition.y + halfScale * m_fieldExtents.y * m_window.getAspectRatio());

    // Ensure bounding box is within field bounds, construct and return it
    topLeft                     = glm::max(topLeft, glm::vec2(0.0f));
    bottomRight                 = glm::min(bottomRight, glm::vec2(m_fieldExtents.x - 1U, m_fieldExtents.y - 1U));
    BoundingBox boundingBox     = {.topLeft = make_uint2(topLeft.x, topLeft.y),
                                   .bottomRight = make_uint2(bottomRight.x, bottomRight.y)};
    return boundingBox;
}

dim3 FieldManager::brushGridDims(BoundingBox boundingBox) {
    return dim3((boundingBox.bottomRight.x - boundingBox.topLeft.x) / utils::BLOCK_SIZE.x,
                (boundingBox.bottomRight.y - boundingBox.topLeft.y) / utils::BLOCK_SIZE.y);
}
