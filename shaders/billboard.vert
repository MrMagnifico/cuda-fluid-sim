#version 460

// Scale and rotation of billboard
layout(location = 0) uniform mat4 transform;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoords;

layout(location = 0) out vec2 fragTexCoords;

void main() {
    fragTexCoords   = texCoords;
    gl_Position     = transform * vec4(position, 1.0);
}
