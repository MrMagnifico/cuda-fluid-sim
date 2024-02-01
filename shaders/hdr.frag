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

#define PI 3.1415926535897932384626433832795

vec3 hsvToRgb(vec3 hsv) {
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;

    h /= 60.0;
    float i = floor(h);
    float f = h - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));

    if (i == 0.0)       { return vec3(v, t, p); }
    else if (i == 1.0)  { return vec3(q, v, p); }
    else if (i == 2.0)  { return vec3(p, v, t); }
    else if (i == 3.0)  { return vec3(p, q, v); }
    else if (i == 4.0)  { return vec3(t, p, v); }
    else                { return vec3(v, p, q); }
}

// Map the angle of the velocity to a hue and its magnitude to a value in HSV.
// Saturation is assumed to be maximal
vec3 velocityVisualisation(vec2 velocity) {
    velocity.y      = -velocity.y;                  // Y grows in the downward direction in CUDA, so we reflect that
    velocity        += vec2(1e-5, 1e-5);            // Avoid division by zero errors
    float angle     = atan(velocity.y, velocity.x);
    float magnitude = length(velocity);
    float hue       = degrees(angle) + 180.0;       // Convert from [-PI, PI] to [0, 360]
    vec3 hsv        = vec3(hue, 1.0, magnitude);    // Use magnitude as unbounded value (tone-mapping should take care of >1 values)
    return hsvToRgb(hsv);
}

void main() {
    // Combine colors from fields textures
    vec3 hdrColor = vec3(0.0);
    if (drawSourcesDensity)     { hdrColor += texture(sourcesDensityTex, bufferCoords).rgb; }
    if (drawSourcesVelocity)    { hdrColor += velocityVisualisation(texture(sourcesVelocityTex, bufferCoords).xy); }
    if (drawDensities)          { hdrColor += texture(densitiesTex, bufferCoords).rgb; }
    if (drawVelocities)         { hdrColor += velocityVisualisation(texture(velocitiesTex, bufferCoords).xy); }
    
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
