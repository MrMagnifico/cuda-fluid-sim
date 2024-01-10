target_sources(FluidLib
	PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/device/field_manager.cu"

        "${CMAKE_CURRENT_LIST_DIR}/render/field_renderer.cpp"

        "${CMAKE_CURRENT_LIST_DIR}/ui/menu.cpp"
        
        "${CMAKE_CURRENT_LIST_DIR}/utils/cuda_utils.cu")
