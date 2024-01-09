#version 460

// HDR rendering params
layout(location = 0) uniform bool hdr;
layout(location = 1) uniform float exposure;
layout(location = 2) uniform float gamma;

// Output fields textures and draw toggles
layout(location = 3) uniform sampler2D sourcesDensityTex;
layout(location = 4) uniform bool drawSourcesDensity;
layout(location = 5) uniform sampler2D sourcesVelocityTex;
layout(location = 6) uniform bool drawSourcesVelocity;
layout(location = 7) uniform sampler2D densitiesTex;
layout(location = 8) uniform bool drawDensities;
layout(location = 9) uniform sampler2D velocitiesTex;
layout(location = 10) uniform bool drawVelocities;

// Quad texture to use with HDR buffer
layout(location = 0) in vec2 bufferCoords;

// Output
layout(location = 0) out vec4 fragColor;

void main() {
    // Combine colors from fields textures
    vec3 hdrColor = vec3(0.0);
    if (drawSourcesDensity)     { hdrColor += texture(sourcesDensityTex, bufferCoords).rgb; }
    if (drawSourcesVelocity)    { hdrColor += texture(sourcesVelocityTex, bufferCoords).rgb; }
    if (drawDensities)          { hdrColor += texture(densitiesTex, bufferCoords).rgb; }
    if (drawVelocities)         { hdrColor += texture(velocitiesTex, bufferCoords).rgb; }
    
    if (hdr) {
        // vec3 result = hdrColor / (hdrColor + vec3(1.0));     // Reinhard tone mapping
        vec3 result = vec3(1.0) - exp(-hdrColor * exposure);    // Exposure tone mapping
        result      = pow(result, vec3(1.0 / gamma));           // Gamma correction
        fragColor   = vec4(result, 1.0);
    }
    else {
        vec3 result = pow(hdrColor, vec3(1.0 / gamma));
        fragColor   = vec4(result, 1.0);
    }
}