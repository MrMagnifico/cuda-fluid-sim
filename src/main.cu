#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <imgui/imgui.h>
DISABLE_WARNINGS_POP()

#include <framework/shader.h>
#include <framework/window.h>

#include <device/fluid_sim.cuh>
#include <device/gl_interop.cuh>
#include <device/sources.cuh>
#include <render/config.h>
#include <ui/callbacks.hpp>
#include <ui/menu.h>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>
#include <utils/render_utils.hpp>

#include <cuda_gl_interop.h>
#include <iostream>


int main(int argc, char* argv[]) {
    // Init core object(s)
    Window m_window("CUDA Fluid Solver", glm::ivec2(utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT), OpenGLVersion::GL46);
    RenderConfig m_renderConfig;
    ui::Menu m_menu(m_renderConfig);

    // Allocate CUDA buffer memory
    const uint2 fieldExtents    = { utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT };
    const size_t fieldsSize     = (utils::INITIAL_WIDTH + 2UL) * (utils::INITIAL_HEIGHT + 2UL) * sizeof(glm::vec3); // Account for ghost cells
    glm::vec3 *densities, *densitiesPrev, *velocities, *velocitiesPrev, *sources;
    CUDA_ERROR(cudaMalloc(&densities, fieldsSize));
    CUDA_ERROR(cudaMalloc(&densitiesPrev, fieldsSize));
    CUDA_ERROR(cudaMalloc(&velocities, fieldsSize));
    CUDA_ERROR(cudaMalloc(&velocitiesPrev, fieldsSize));
    CUDA_ERROR(cudaMalloc(&sources, fieldsSize));

    // Zero out fields
    CUDA_ERROR(cudaMemset(densities,        0, fieldsSize));
    CUDA_ERROR(cudaMemset(densitiesPrev,    0, fieldsSize));
    CUDA_ERROR(cudaMemset(velocities,       0, fieldsSize));
    CUDA_ERROR(cudaMemset(velocitiesPrev,   0, fieldsSize));
    CUDA_ERROR(cudaMemset(sources,          0, fieldsSize));

    // Set sources
    const uint2 paddedfieldExtents = { utils::INITIAL_WIDTH + 2U, utils::INITIAL_HEIGHT + 2U };
    for (unsigned int i = 0U; i < 10U; i++) {
        for (unsigned int j = 0U; j < 10U; j++) {
            set_source<<<1, 1>>>(sources, make_uint2(300U + i, 300U + j), glm::vec3(0.2f, 0.0f, 0.0f), paddedfieldExtents);
            set_source<<<1, 1>>>(sources, make_uint2(600U + i, 400U + j), glm::vec3(0.0f, 0.2f, 0.0f), paddedfieldExtents);
            set_source<<<1, 1>>>(sources, make_uint2(500U + i, 400U + j), glm::vec3(0.0f, 0.0f, 0.2f), paddedfieldExtents);
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    // Determine grid dimensions needed for workload distribution
    dim3 gridDims((paddedfieldExtents.x / utils::BLOCK_SIZE.x) + 1U, (paddedfieldExtents.y / utils::BLOCK_SIZE.y) + 1U);

    // Compile quad render w/ HDR shader program
    ShaderBuilder quadHdrBuilder;
    quadHdrBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "screen-quad.vert");
    quadHdrBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "hdr.frag");
    Shader quadHdr = quadHdrBuilder.build();

    // Init OpenGL textures and bind them to CUDA resources
    GLuint densitiesTex, velocitiesTex;
    cudaGraphicsResource_t densitiesResource, velocitiesResource;
    glCreateTextures(GL_TEXTURE_2D, 1, &densitiesTex);
    glTextureStorage2D(densitiesTex, 1, GL_RGBA32F, fieldExtents.x, fieldExtents.y);
    glTextureParameteri(densitiesTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(densitiesTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, densitiesTex);
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&densitiesResource, densitiesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    glCreateTextures(GL_TEXTURE_2D, 1, &velocitiesTex);
    glTextureStorage2D(velocitiesTex, 1, GL_RGBA32F, fieldExtents.x, fieldExtents.y);
    glTextureParameteri(velocitiesTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(velocitiesTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, velocitiesTex);
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&velocitiesResource, velocitiesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register UI callbacks
    m_window.registerKeyCallback(ui::keyCallback);
    m_window.registerMouseMoveCallback(ui::onMouseMove);
    m_window.registerMouseButtonCallback(ui::mouseButtonCallback);

    // Main loop
    while (!m_window.shouldClose()) {
        // Main simulation
        add_sources<<<gridDims, utils::BLOCK_SIZE>>>(densities, sources, m_renderConfig.timeStep, paddedfieldExtents.x * paddedfieldExtents.y);
        std::swap(densities, densitiesPrev);
        diffuse<<<gridDims, utils::BLOCK_SIZE>>>(densitiesPrev, densities, fieldExtents, paddedfieldExtents.x * paddedfieldExtents.y,
                                                 m_renderConfig.timeStep, m_renderConfig.diffusionRate, m_renderConfig.diffusionSimSteps);

        // Write fields to OpenGL textures
        cudaSurfaceObject_t densitiesSurface    = utils::createSurfaceFromTextureResource(densitiesResource);
        cudaSurfaceObject_t velocitiesSurface   = utils::createSurfaceFromTextureResource(velocitiesResource);
        copyFieldToTexture<<<gridDims, utils::BLOCK_SIZE>>>(densities, densitiesSurface, fieldExtents);
        copyFieldToTexture<<<gridDims, utils::BLOCK_SIZE>>>(velocities, velocitiesSurface, fieldExtents);
        CUDA_ERROR(cudaDeviceSynchronize()); // Ensure that copying is over before terminating resource handles
        CUDA_ERROR(cudaDestroySurfaceObject(densitiesSurface));
        CUDA_ERROR(cudaDestroySurfaceObject(velocitiesSurface));
        CUDA_ERROR(cudaGraphicsUnmapResources(1, &densitiesResource));
        CUDA_ERROR(cudaGraphicsUnmapResources(1, &velocitiesResource));

        // Clear the screen
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Process controls
        ImGuiIO io = ImGui::GetIO();
        m_window.updateInput();
        if (!io.WantCaptureMouse) { /* Non ImGUI UI code */ }

        // Bind render shader, set uniforms, and render
        quadHdr.bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, densitiesTex);
        glUniform1i(0, 0);
        glUniform1i(1, m_renderConfig.renderDensities);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, velocitiesTex);
        glUniform1i(2, 1);
        glUniform1i(3, m_renderConfig.renderVelocities);
        glUniform1i(4, m_renderConfig.enableHdr);
        glUniform1f(5, m_renderConfig.exposure);
        glUniform1f(6, m_renderConfig.gamma);
        utils::renderQuad();
        glBindTexture(GL_TEXTURE_2D, 0);

        // Draw UI
        m_menu.draw();

        // Processes input and swaps the window buffer
        m_window.swapBuffers();
    }

    return EXIT_SUCCESS;
}
