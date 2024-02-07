#include "texture.h"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <fmt/format.h>
DISABLE_WARNINGS_POP()

#include <framework/image.h>

#include <exception>
#include <iostream>

void Texture::bind(GLint textureSlot) const {
    glActiveTexture(textureSlot);
    glBindTexture(GL_TEXTURE_2D, m_texture);
}

std::weak_ptr<const Texture> TextureManager::addTexture(std::filesystem::path filePath) {
    // Declare texture and load image from disk to CPU memory
    Texture newTexture;
    Image cpuTexture(filePath);

    // Manually copy over string of filename
    std::string tempName = filePath.filename().string();
    newTexture.fileName.resize(tempName.length());
    tempName.copy(newTexture.fileName.data(), tempName.length(), 0);

    // Create texture with correct format, upload data, and generate mipmaps
    glCreateTextures(GL_TEXTURE_2D, 1, &newTexture.m_texture);
    switch (cpuTexture.channels) {
        case 1:
            glTextureStorage2D(newTexture.m_texture, 1, GL_R8, cpuTexture.width, cpuTexture.height);
            glTextureSubImage2D(newTexture.m_texture, 0, 0, 0, cpuTexture.width, cpuTexture.height, GL_RED, GL_UNSIGNED_BYTE, cpuTexture.pixels.data());
            break;
        case 3:
            glTextureStorage2D(newTexture.m_texture, 1, GL_RGB8, cpuTexture.width, cpuTexture.height);
            glTextureSubImage2D(newTexture.m_texture, 0, 0, 0, cpuTexture.width, cpuTexture.height, GL_RGB, GL_UNSIGNED_BYTE, cpuTexture.pixels.data());
            break;
        case 4:
            glTextureStorage2D(newTexture.m_texture, 1, GL_RGBA8, cpuTexture.width, cpuTexture.height);
            glTextureSubImage2D(newTexture.m_texture, 0, 0, 0, cpuTexture.width, cpuTexture.height, GL_RGBA, GL_UNSIGNED_BYTE, cpuTexture.pixels.data());
            break;
        default:
            std::cerr << "Number of channels read for texture is not supported" << std::endl;
            throw std::exception();
    }
    glGenerateTextureMipmap(newTexture.m_texture);

    // Set behavior for when texture coordinates are outside the [0, 1] range.
    glTextureParameteri(newTexture.m_texture, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(newTexture.m_texture, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Set interpolation for texture sampling (mipmaps)
    glTextureParameteri(newTexture.m_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(newTexture.m_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create shared pointer managing the new texture and add it
    textures.push_back(std::make_shared<Texture>(newTexture));
    return textures.back();
}

bool TextureManager::removeTexture(std::string fileName) {
    for (size_t idx = 0UL; idx < textures.size(); idx++) {
        std::shared_ptr<Texture> tex = textures[idx];
        if (fileName == tex->fileName) {
            glDeleteTextures(1, &tex->m_texture);
            textures.erase(textures.begin() + idx);
            return true;
        }
    }
    return false;
}

std::weak_ptr<const Texture> TextureManager::getTexture(std::string fileName) const {
    for (const std::shared_ptr<Texture> tex : textures) { if (fileName == tex->fileName) { return tex; } }
    return std::weak_ptr<Texture>();
}
