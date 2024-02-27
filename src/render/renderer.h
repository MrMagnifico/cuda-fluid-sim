#ifndef _RENDERER_H_
#define _RENDERER_H_


#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <glm/vec2.hpp>
DISABLE_WARNINGS_POP()

#include <framework/shader.h>
#include <framework/window.h>
#include <render/config.h>
#include <render/texture.h>


class Renderer {
    public:
        Renderer(const RenderConfig& renderConfig, const Window& window,
                      std::weak_ptr<const Texture> brushTex,
                      unsigned int fieldWidth, unsigned int fieldHeight);
        ~Renderer();

        void renderFields();
        void renderBrush();
        void resizeTextures(unsigned int fieldWidth, unsigned int fieldHeight);

        GLuint getSourcesDensityTex() const     { return m_sourcesDensityTex; }
        GLuint getDensitiesTex() const          { return m_densitiesTex; }
        GLuint getSourcesVelocityTex() const    { return m_sourcesVelocityTex; }
        GLuint getVelocitiesTex() const         { return m_velocitiesTex; }

    private:
        const RenderConfig& m_renderConfig;
        const Window& m_window;

        glm::uvec2 m_fieldDims;
        GLuint m_sourcesDensityTex, m_densitiesTex, m_sourcesVelocityTex, m_velocitiesTex;
        std::weak_ptr<const Texture> m_brushTex;
        Shader m_brushBillboard, m_quadHdr;

        void initShaderPrograms();
        void initTextures();
        void destroyTextures();
};


#endif
