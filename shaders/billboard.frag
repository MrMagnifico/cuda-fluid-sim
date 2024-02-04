#version 460

layout(location = 1) uniform sampler2D brushTex;

layout(location = 0) in vec2 texCoords;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(brushTex, texCoords);
}
