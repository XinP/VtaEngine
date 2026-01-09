cmake_minimum_required(VERSION 3.10)
project(onnx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to ONNX Runtime
# Using relative path assuming the folder is in the project root
set(ONNX_RUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2")
set(ONNX_INCLUDE_DIR "${ONNX_RUNTIME_DIR}/include")
set(ONNX_LIB_DIR "${ONNX_RUNTIME_DIR}/lib")

# Include directories
include_directories(${ONNX_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    onnx.cpp
    OnnxInfer.cpp
    src/config/LeadConfig.cpp
)

# Header files (optional, but good for IDEs)
set(HEADERS
    OnnxInfer.h
    include/config/LeadConfig.h
    include/common/Types.h
    include/common/MathUtils.h
)

add_executable(onnx ${SOURCES} ${HEADERS})

# Link libraries
# Note: linking against .lib files from the precompiled Windows package
target_link_libraries(onnx 
    "${ONNX_LIB_DIR}/onnxruntime.lib"
    "${ONNX_LIB_DIR}/onnxruntime_providers_shared.lib"
)

# Post-build commands
# 1. Copy DLLs to the output directory
# 2. Copy models directory to the output directory
add_custom_command(TARGET onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNX_LIB_DIR}/onnxruntime.dll"
    $<TARGET_FILE_DIR:onnx>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNX_LIB_DIR}/onnxruntime_providers_shared.dll"
    $<TARGET_FILE_DIR:onnx>
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:onnx>/models"
    COMMENT "Copying runtime DLLs and models to output directory"
)
