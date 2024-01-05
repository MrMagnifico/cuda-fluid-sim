#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoords;

layout(location = 0) out vec2 bufferCoords;

void main() {
    bufferCoords    = texCoords;
    gl_Position     = vec4(position, 1.0);
}
