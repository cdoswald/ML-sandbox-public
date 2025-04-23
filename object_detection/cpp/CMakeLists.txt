cmake_minimum_required(VERSION 3.10)
project(ObjectDetectionApp)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
find_library(ONNX_RUNTIME_LIB onnxruntime PATHS /opt/onnxruntime/lib)

# Add source file
add_executable(main /workspace/main.cpp)

# Include OpenCV headers
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(main ${OpenCV_LIBS} ${ONNX_RUNTIME_LIB})