cmake_minimum_required(VERSION 3.25)
project(vio-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================
# 1. Dependencies
# ==========================================
find_package(Eigen3 5.0 REQUIRED)
find_package(Ceres 2.0 REQUIRED)
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs video)
find_package(yaml-cpp REQUIRED)

# Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(VISION_FRAMEWORK Vision REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ==========================================
# 2. Metal Shader Compilation
# ==========================================
set(METAL_SHADERS
    src/metal/shaders/undistort.metal
)
foreach(shader ${METAL_SHADERS})
    get_filename_component(shader_name ${shader} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${shader_name}.metallib
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/${shader}
                -o ${CMAKE_BINARY_DIR}/${shader_name}.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/${shader_name}.air
                -o ${CMAKE_BINARY_DIR}/${shader_name}.metallib
        DEPENDS ${shader}
    )
    list(APPEND METALLIB_OUTPUTS ${CMAKE_BINARY_DIR}/${shader_name}.metallib)
endforeach()
add_custom_target(metal_shaders DEPENDS ${METALLIB_OUTPUTS})

# ==========================================
# 3. Main Executable (Full Pipeline)
# ==========================================
add_executable(vio-metal
    src/main.mm
    src/dataset/EurocLoader.cpp
    src/dataset/TrajectoryWriter.cpp
    src/metal/MetalContext.mm
    src/metal/MetalUndistort.mm
    src/vision/FeatureDetector.cpp
    src/vision/StereoMatcher.cpp
    src/vision/TemporalTracker.mm
    src/vision/FeatureManager.cpp
    src/imu/ImuPreintegrator.cpp
    src/optimization/VioOptimizer.cpp
    src/optimization/Factors.cpp
    src/optimization/Marginalization.cpp
    src/core/KeyframePolicy.cpp
    src/core/Profiler.cpp
)
# Enable ARC for Objective-C++ files
set_source_files_properties(src/main.mm src/metal/MetalContext.mm src/metal/MetalUndistort.mm src/vision/TemporalTracker.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

add_dependencies(vio-metal metal_shaders)

target_include_directories(vio-metal PRIVATE
    src/
    ${EIGEN3_INCLUDE_DIR}
)

# Main executable gets ALL dependencies
target_link_libraries(vio-metal PRIVATE
    Eigen3::Eigen
    ceres
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    ${VISION_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Eigen → Accelerate BLAS
target_compile_definitions(vio-metal PRIVATE EIGEN_USE_BLAS)

# ==========================================
# 4. Tests
# ==========================================

# Test Executable: EuRoC Loader
add_executable(test_euroc_loader
    test/test_euroc_loader.cpp
    src/dataset/EurocLoader.cpp
)

target_include_directories(test_euroc_loader PRIVATE
    src/
    ${EIGEN3_INCLUDE_DIR}
)

# Test executable ONLY gets what it needs (OpenCV, Eigen, Yaml)
target_link_libraries(test_euroc_loader PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
)