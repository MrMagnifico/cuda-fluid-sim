#ifndef _MENU_H_
#define _MENU_H_

#include <render/config.h>

namespace ui {
    class Menu {
    public:
        Menu(RenderConfig& renderConfig);

        void draw();

    private:
        RenderConfig& m_renderConfig;

        void drawSimControlsTab();
        void drawRenderTab();
        void drawBrushTab();

        void drawSimParamsControls();
        void drawSimTogglesControls();

        void drawFieldDrawControls();        
        void drawHdrControls();
        void drawMiscControls();
    };

}

#endif
