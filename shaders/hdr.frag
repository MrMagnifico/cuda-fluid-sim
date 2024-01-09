#version 460

// Output fields textures and draw toggles
layout(location = 0) uniform sampler2D densitiesTex;
layout(location = 1) uniform bool drawDensities;
layout(location = 2) uniform sampler2D sourcesTex;
layout(location = 3) uniform bool drawSources;

// HDR rendering params
layout(location = 4) uniform bool hdr;
layout(location = 5) uniform float exposure;
layout(location = 6) uniform float gamma;

// Quad texture to use with HDR buffer
layout(location = 0) in vec2 bufferCoords;

// Output
layout(location = 0) out vec4 fragColor;

void main() {
    // Combine colors from fields textures
    vec3 hdrColor = vec3(0.0);
    if (drawDensities)  { hdrColor += texture(densitiesTex, bufferCoords).rgb; }
    if (drawSources)    { hdrColor += texture(sourcesTex, bufferCoords).rgb; }

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