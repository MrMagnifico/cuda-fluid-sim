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
    std::array<GLuint, 2UL> textures = { m_densitiesTex, m_sourcesTex };
    glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
}

void FieldRenderer::initShaderProgram() {
    ShaderBuilder quadHdrBuilder;
    quadHdrBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "screen-quad.vert");
    quadHdrBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "hdr.frag");
    m_quadHdr = quadHdrBuilder.build();   
}

void FieldRenderer::initTextures() {
    std::array<GLuint*, 2UL> textures = { &m_densitiesTex, &m_sourcesTex };
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
    m_quadHdr.bind();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_densitiesTex);
    glUniform1i(0, 0);
    glUniform1i(1, m_renderConfig.renderDensities);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_sourcesTex);
    glUniform1i(2, 1);
    glUniform1i(3, m_renderConfig.renderSources);
    glUniform1i(4, m_renderConfig.enableHdr);
    glUniform1f(5, m_renderConfig.exposure);
    glUniform1f(6, m_renderConfig.gamma);
    utils::renderQuad();
    glBindTexture(GL_TEXTURE_2D, 0);
}
