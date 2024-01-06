target_sources(FluidLib
	PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/device/fluid_sim.cu"
        "${CMAKE_CURRENT_LIST_DIR}/device/gl_interop.cu"
        "${CMAKE_CURRENT_LIST_DIR}/device/sources.cu"

        "${CMAKE_CURRENT_LIST_DIR}/ui/menu.cpp")
