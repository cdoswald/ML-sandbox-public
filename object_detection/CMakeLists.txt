cmake_minimum_required(VERSION 3.10)
project(ObjectDetectionApp)
set(CMAKE_CXX_STANDARD 17)

# Find ONNX Runtime
find_library(ONNX_RUNTIME_LIB onnxruntime PATHS /opt/onnxruntime/lib)

# Add source file
add_executable(main /workspace/main.cpp)

# Link ONNX Runtime library
target_link_libraries(main ${ONNX_RUNTIME_LIB})