#include "menu.h"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>
#include <nativefiledialog/nfd.h>
DISABLE_WARNINGS_POP()

#include <filesystem>
#include <iostream>


void ui::Menu::draw() {
    ImGui::Begin("Controls");
    ImGui::End();
}
