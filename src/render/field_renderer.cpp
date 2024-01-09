#include "field_renderer.h"

#include <utils/constants.h>
#include <utils/render_utils.hpp>

#include <array>

FieldRenderer::FieldRenderer(const RenderConfig& renderConfig, unsigned int fieldWidth, unsigned int fieldHeight)
    : m_renderConfig(renderConfig)
    , m_fieldDims(fieldWidth, fieldHeight) {
    initShaderProgram();
    initTextures();
}

FieldRenderer::~FieldRenderer() {
    std::array<GLuint, 4UL> textures = { m_sourcesDensityTex, m_densitiesTex, m_sourcesVelocityTex, m_velocitiesTex };
    glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
}

void FieldRenderer::initShaderProgram() {
    ShaderBuilder quadHdrBuilder;
    quadHdrBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "screen-quad.vert");
    quadHdrBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "hdr.frag");
    m_quadHdr = quadHdrBuilder.build();   
}

void FieldRenderer::initTextures() {
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

void FieldRenderer::render() {
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
