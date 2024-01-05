#pragma once

// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
DISABLE_WARNINGS_POP()

#include <array>
#include <filesystem>
#include <stdint.h>
#include <vector>

struct Image {
public:
    Image(const std::filesystem::path& filePath);

    std::vector<uint8_t> getTexel(const glm::vec2& textureCoordinates) const;

    int32_t width, height, channels;
    std::vector<uint8_t> pixels;
};
