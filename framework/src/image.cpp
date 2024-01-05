#include "image.h"
// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
DISABLE_WARNINGS_POP()
#include <cassert>
#include <exception>
#include <iostream>
#include <string>

Image::Image(const std::filesystem::path& filePath) {
	if (!std::filesystem::exists(filePath)) {
		std::cerr << "Texture file " << filePath << " does not exist!" << std::endl;
		throw std::exception();
	}

	const auto filePathStr = filePath.string(); // Create l-value so c_str() is safe.
	stbi_uc* stbPixels = stbi_load(filePathStr.c_str(), &width, &height, &channels, STBI_default);

	if (!stbPixels) {
		std::cerr << "Failed to read texture " << filePath << " using stb_image.h" << std::endl;
		throw std::exception();
	}

	for (size_t i = 0UL; i < width * height * channels; i++) { pixels.emplace_back(stbPixels[i]); }

	stbi_image_free(stbPixels);
}

std::vector<uint8_t> Image::getTexel(const glm::vec2& textureCoordinates) const {
	const glm::ivec2 pixel 		= glm::ivec2(textureCoordinates * glm::vec2(width, height) + 0.5f);
	const size_t pixelOffset	= pixel.y * width + pixel.x;

	// Return appropriate number of channels
	std::vector<uint8_t> pixelData;
	for (size_t i = pixelOffset; i < pixelOffset + channels; i++) { pixelData.push_back(pixels[i]); }
	return pixelData;
}
