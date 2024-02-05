#include "menu.h"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>
#include <nativefiledialog/nfd.h>
DISABLE_WARNINGS_POP()

#include <utils/constants.h>
#include <utils/magic_enum.hpp>

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
    drawBrushTab();

    ImGui::EndTabBar();
    ImGui::End();
}

void ui::Menu::drawSimControlsTab() {
    if (ImGui::BeginTabItem("Simulation")) {
        ImGui::Text("Step Toggles");
        drawSimTogglesControls();
        
        ImGui::NewLine();
        ImGui::Separator();
        
        ImGui::Text("Parameters");
        drawSimParamsControls();

        ImGui::EndTabItem();
    }
}

void ui::Menu::drawSimTogglesControls() {
    ImGui::Text("Densities");
    ImGui::Checkbox("Add sources##Densities",   &m_renderConfig.densityAddSources);
    ImGui::Checkbox("Diffuse##Densities",       &m_renderConfig.densityDiffuse);
    ImGui::Checkbox("Advect##Densities",        &m_renderConfig.densityAdvect);
    ImGui::NewLine();
    ImGui::Text("Velocities");
    ImGui::Checkbox("Add sources##Velocities",  &m_renderConfig.velocityAddSources);
    ImGui::Checkbox("Diffuse##Velocities",      &m_renderConfig.velocityDiffuse);
    ImGui::Checkbox("Advect##Velocities",       &m_renderConfig.velocityAdvect);
    ImGui::Checkbox("Project##Velocities",      &m_renderConfig.velocityProject);
}

void ui::Menu::drawSimParamsControls() {
    ImGui::InputFloat("Time step", &m_renderConfig.simulationParams.timeStep, 0.001f, 0.01f, "%.3f");
    ImGui::InputFloat("Diffusion rate", &m_renderConfig.simulationParams.diffusionRate, 1.0f, 10.0f, "%.0f");
    ImGui::SliderInt("Diffusion simulation steps", reinterpret_cast<int*>(&m_renderConfig.simulationParams.diffusionSimSteps), 1, 64);
    ImGui::InputFloat("Advection multiplier", &m_renderConfig.simulationParams.advectionMultiplier, 0.1f, 1.0f, "%.1f");
    ImGui::SliderInt("Projection simulation steps", reinterpret_cast<int*>(&m_renderConfig.simulationParams.projectionSimSteps), 1, 64);
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
    ImGui::Checkbox("Sources - Densities", &m_renderConfig.renderDensitySources);
    ImGui::Checkbox("Sources - Velocities", &m_renderConfig.renderVelocitySources);
    ImGui::Checkbox("Densities", &m_renderConfig.renderDensities);
    ImGui::Checkbox("Velocities", &m_renderConfig.renderVelocities);
}

void ui::Menu::drawHdrControls() {
    ImGui::Checkbox("Enable HDR", &m_renderConfig.enableHdr);
    ImGui::InputFloat("Exposure", &m_renderConfig.exposure, 0.1f, 1.0f, "%.1f");
    ImGui::InputFloat("Gamma", &m_renderConfig.gamma, 0.1f, 1.0f, "%.1f");
}

void ui::Menu::drawBrushTab() {
    if (ImGui::BeginTabItem("Brush")) {
        // Options for which field to affect
        constexpr auto fieldOptions = magic_enum::enum_names<BrushEditMode>();
        std::vector<const char*> fieldOptionsPointers;
        std::transform(std::begin(fieldOptions), std::end(fieldOptions), std::back_inserter(fieldOptionsPointers),
            [](const auto& str) { return str.data(); });

        ImGui::SliderFloat("Size", &m_renderConfig.brushParams.scale, utils::MIN_BRUSH_SIZE, utils::MAX_BRUSH_SIZE, "%.2f");
        ImGui::Combo("Field to paint", (int*) &m_renderConfig.brushParams.brushEditMode, fieldOptionsPointers.data(), static_cast<int>(fieldOptionsPointers.size()));
        ImGui::ColorEdit3("Density brush colour", glm::value_ptr(m_renderConfig.brushParams.densityDrawColor));\
        ImGui::DragFloat2("Velocity brush value", glm::value_ptr(m_renderConfig.brushParams.velocityDrawValue), 0.01f, -10.0f, 10.0f, "%.2f");
        ImGui::EndTabItem();
    }
}
