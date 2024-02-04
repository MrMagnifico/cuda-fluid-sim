#include "renderer.h"

DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
DISABLE_WARNINGS_POP()


#include <utils/constants.h>
#include <utils/render_utils.hpp>
#include <utils/misc_utils.hpp>

#include <array>

Renderer::Renderer(const RenderConfig& renderConfig, const Window& window,
                             std::weak_ptr<const Texture> brushTex,
                             unsigned int fieldWidth, unsigned int fieldHeight)
    : m_renderConfig(renderConfig)
    , m_window(window)
    , m_brushTex(brushTex)
    , m_fieldDims(fieldWidth, fieldHeight) {
    initShaderPrograms();
    initTextures();
}

Renderer::~Renderer() {
    std::array<GLuint, 4UL> textures = { m_sourcesDensityTex, m_densitiesTex, m_sourcesVelocityTex, m_velocitiesTex };
    glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
}

void Renderer::initShaderPrograms() {
    ShaderBuilder quadHdrBuilder;
    quadHdrBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "screen-quad.vert");
    quadHdrBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "hdr.frag");
    m_quadHdr = quadHdrBuilder.build();

    ShaderBuilder brushBillboardBuilder;
    brushBillboardBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "billboard.vert");
    brushBillboardBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "billboard.frag");
    m_brushBillboard = brushBillboardBuilder.build();
}

void Renderer::initTextures() {
    std::array<GLuint*, 4UL> textures = { &m_sourcesDensityTex, &m_densitiesTex, &m_sourcesVelocityTex, &m_velocitiesTex };
    for (GLuint* texture : textures) {
        glCreateTextures(GL_TEXTURE_2D, 1, texture);
        glTextureStorage2D(*texture, 1, GL_RGBA32F, m_fieldDims.x, m_fieldDims.y);
        glTextureParameteri(*texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(*texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, *texture);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::renderFields() {
    // Bind shader program and set HDR params
    m_quadHdr.bind();
    glUniform1i(0, m_renderConfig.enableHdr);
    glUniform1f(1, m_renderConfig.exposure);
    glUniform1f(2, m_renderConfig.gamma);

    // Configure texture samplers and drawing toggles
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_sourcesDensityTex);
    glUniform1i(3, 0);
    glUniform1i(4, m_renderConfig.renderDensitySources);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_sourcesVelocityTex);
    glUniform1i(5, 1);
    glUniform1i(6, m_renderConfig.renderVelocitySources);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_densitiesTex);
    glUniform1i(7, 2);
    glUniform1i(8, m_renderConfig.renderDensities);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_velocitiesTex);
    glUniform1i(9, 3);
    glUniform1i(10, m_renderConfig.renderVelocities);
    
    // Render fullscreen quad and release texture bind
    utils::renderQuad();
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::renderBrush() {
    // Compute transformation to cursor position at desired size
    glm::vec2 cursorPos     = m_window.getNormalizedCursorPos();
    cursorPos               = utils::mapRange(cursorPos, glm::vec2(0.0f), glm::vec2(1.0f),
                                                         glm::vec2(-1.0f), glm::vec2(1.0f));                    // Transform cursor position to OpenGL's [-1, 1] range
    glm::mat4 viewportScale = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, m_window.getAspectRatio(), 1.0f));    // Full-screen quad is rendered at screen's aspect ratio by default. This undoes that
    glm::mat4 scale         = glm::scale(glm::mat4(1.0f), glm::vec3(m_renderConfig.brushParams.scale));         // Scale brush to desired size
    glm::mat4 translate     = glm::translate(glm::mat4(1.0f), glm::vec3(cursorPos.x, cursorPos.y, 0.0f));       // Translate to cursor position
    glm::mat4 transform     = translate * scale * viewportScale;

    // Bind shader program and set uniforms for transformation matrix and billboard texture sampler
    m_brushBillboard.bind();
    glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(transform));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_brushTex.lock().get()->m_texture);
    glUniform1i(1, 0);
    
    // Render fullscreen quad and release texture bind
    utils::renderQuad();
    glBindTexture(GL_TEXTURE_2D, 0);
}
