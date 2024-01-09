#ifndef _FIELD_RENDERER_H_
#define _FIELD_RENDERER_H_


#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <glm/vec2.hpp>
DISABLE_WARNINGS_POP()

#include <render/config.h>
#include <framework/shader.h>


class FieldRenderer {
    public:
        FieldRenderer(const RenderConfig& renderConfig, unsigned int fieldWidth, unsigned int fieldHeight);
        ~FieldRenderer();

        void render();

        GLuint getSourcesDensityTex() const     { return m_sourcesDensityTex; }
        GLuint getDensitiesTex() const          { return m_densitiesTex; }
        GLuint getSourcesVelocityTex() const    { return m_sourcesVelocityTex; }
        GLuint getVelocitiesTex() const         { return m_velocitiesTex; }

    private:
        const RenderConfig& m_renderConfig;

        glm::uvec2 m_fieldDims;
        GLuint m_sourcesDensityTex, m_densitiesTex, m_sourcesVelocityTex, m_velocitiesTex;
        Shader m_quadHdr;

        void initShaderProgram();
        void initTextures();
};


#endif
