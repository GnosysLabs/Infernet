if(NOT PROJECT_NAME STREQUAL "stable-diffusion")
    return()
endif()

if(NOT DEFINED INFERNET_IMAGE_RPC_SOURCE_DIR)
    message(FATAL_ERROR "INFERNET_IMAGE_RPC_SOURCE_DIR is required")
endif()

# CMake resolves the `ggml` link target during generation, after the upstream
# project has declared it. This keeps Infernet's small server wrapper outside
# the immutable checkout while sharing sd-cli's exact GGML core.
add_subdirectory(
    "${INFERNET_IMAGE_RPC_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/infernet-image-rpc"
)
