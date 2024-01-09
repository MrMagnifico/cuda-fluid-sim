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

        const GLuint getSourcesTex() const      { return m_sourcesTex; }
        const GLuint getDensitiesTex() const    { return m_densitiesTex; }

    private:
        const RenderConfig& m_renderConfig;

        // TODO: Expand to other fields
        glm::uvec2 m_fieldDims;
        GLuint m_sourcesTex, m_densitiesTex;
        Shader m_quadHdr;

        void initShaderProgram();
        void initTextures();
};


#endif
