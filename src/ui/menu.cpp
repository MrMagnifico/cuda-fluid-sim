#include "menu.h"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>
#include <nativefiledialog/nfd.h>
DISABLE_WARNINGS_POP()

#include <filesystem>
#include <iostream>

ui::Menu::Menu(RenderConfig& renderConfig)
    : m_renderConfig(renderConfig)
    {}

void ui::Menu::draw() {
    ImGui::Begin("Debug Controls");
    ImGui::BeginTabBar("Controls");
    
    drawSimControlsTab();
    drawRenderTab();

    ImGui::EndTabBar();
    ImGui::End();
}

void ui::Menu::drawSimControlsTab() {
    if (ImGui::BeginTabItem("Simulation Controls")) {
        ImGui::InputFloat("Time step", &m_renderConfig.timeStep, 0.01f, 10.0f, "%.2f");
        ImGui::InputFloat("Diffusion rate", &m_renderConfig.diffusionRate, 0.01f, 10.0f, "%.2f");
        ImGui::SliderInt("Diffusion simulation steps", reinterpret_cast<int*>(&m_renderConfig.diffusionSimSteps), 1, 128);
        ImGui::EndTabItem();
    }
}

void ui::Menu::drawRenderTab() {
    if (ImGui::BeginTabItem("Render")) {
        ImGui::Text("Field Draw");
        drawFieldDrawControls();

        ImGui::NewLine();
        ImGui::Separator();

        ImGui::Text("HDR");
        drawHdrControls();

        ImGui::EndTabItem();
    }
}

void ui::Menu::drawFieldDrawControls() {
    ImGui::Checkbox("Densities", &m_renderConfig.renderDensities);
    ImGui::Checkbox("Velocities", &m_renderConfig.renderVelocities);
}

void ui::Menu::drawHdrControls() {
    ImGui::Checkbox("Enable HDR", &m_renderConfig.enableHdr);
    ImGui::InputFloat("Exposure", &m_renderConfig.exposure, 0.1f, 1.0f, "%.1f");
    ImGui::InputFloat("Gamma", &m_renderConfig.gamma, 0.1f, 1.0f, "%.1f");
}
