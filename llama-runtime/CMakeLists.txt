set(TARGET infernet-llama-bridge)

add_executable(${TARGET} infernet-bridge.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama ggml)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
