add_executable(infernet-image-rpc-server rpc-server.cpp)
target_link_libraries(infernet-image-rpc-server PRIVATE ggml)
set_target_properties(infernet-image-rpc-server PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(
    infernet-image-rpc-server
    PRIVATE INFERNET_SD_RPC_REVISION="cc734292286f85f9c48305d94d7fd22f42838522"
)
