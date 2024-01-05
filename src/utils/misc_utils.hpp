#ifndef _MISC_UTILS_HPP_
#define _MISC_UTILS_HPP_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

#include <random>
#include <vector>

namespace utils {
    /********** Random number generation **********/
    static std::random_device randomDevice;                 // Will be used to obtain a seed for the random number engine
    static std::mt19937 randomGenerator(randomDevice());    // Standard mersenne_twister_engine seeded with randomDevice

    template<typename T>
    static T randomNumber(T low, T high) { 
        std::uniform_real_distribution<T> distribution(low, high);
        return distribution(randomGenerator);
    }

    template<typename T>
    static T acceleratingLinearInterpolation(T low, T high, T interpolant) { return low + interpolant * (high - low); }

    static std::vector<glm::vec4> randomVectors(const glm::vec2& xBounds, const glm::vec2& yBounds, const glm::vec2& zBounds, size_t numSamples,
                                                bool scaleMagnitude = true, bool acceleratingInterpolationScale = true,
                                                const glm::vec2& magnitudeBounds = glm::vec2(0.0f, 1.0f)) {
        std::vector<glm::vec4> generatedVectors(numSamples, {0.0f, 0.0f, 0.0f, 0.0f});
        std::uniform_real_distribution<float> xDistribution(xBounds.x, xBounds.y);
        std::uniform_real_distribution<float> yDistribution(yBounds.x, yBounds.y);
        std::uniform_real_distribution<float> zDistribution(zBounds.x, zBounds.y);
        std::uniform_real_distribution<float> boundsDistribution(zBounds.x, zBounds.y);

        for (size_t sampleIdx = 0UL; sampleIdx < numSamples; sampleIdx++) {
            // Generate random vector using given bounds
            glm::vec4& sample   = generatedVectors[sampleIdx];
            sample.x            = xDistribution(randomGenerator);
            sample.y            = yDistribution(randomGenerator);
            sample.z            = zDistribution(randomGenerator);

            // Set magnitude to given bounds if needed
            if (scaleMagnitude) {
                sample  = glm::normalize(sample);
                sample  *= boundsDistribution(randomGenerator);
            }            

            // Scale samples closer to origin if acceleration interpolation is enabled
            if (acceleratingInterpolationScale) {
                float scale = static_cast<float>(sampleIdx) / static_cast<float>(numSamples);
                sample *= acceleratingLinearInterpolation(0.1f, 1.0f, scale * scale);
            }
        }

        return generatedVectors;
    }

    /********** Geometric utilities **********/
    static glm::vec4 quaternionToAxisAndDegrees(const glm::vec4& quaternion) {
        glm::vec4 in        = glm::normalize(quaternion);
        float angleRads     = acos(in.w) * 2.0f;
        float angleDegrees  = angleRads * 180.0f / glm::pi<float>();
        float x = in.x / sin(angleRads / 2.0f);
        float y = in.y / sin(angleRads / 2.0f);
        float z = in.z / sin(angleRads / 2.0f);
        return glm::vec4(x, y, z, angleDegrees);
    }

    static float eulerDistIgnoreW(glm::vec4 a, glm::vec4 b) { return sqrtf(pow(a.x - b.x, 2.0f) + pow(a.y - b.y, 2.0f) + pow(a.z - b.z, 2.0f)); }
}

#endif
