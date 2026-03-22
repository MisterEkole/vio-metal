cmake_minimum_required(VERSION 3.25)
project(vio-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

# Dependencies
find_package(Eigen3 5.0 REQUIRED)
find_package(Ceres 2.0 REQUIRED)
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs video)
find_package(yaml-cpp REQUIRED)
find_package(Pangolin REQUIRED) 

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(VISION_FRAMEWORK Vision REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Metal Shader Compilation (Compiles to .air, then links to a single shaders.metallib)
set(METAL_SHADERS
    src/metal/shaders/undistort.metal
    src/metal/shaders/FastDetect.metal
    src/metal/shaders/KLTTracker.metal
    src/metal/shaders/StereoMatch.metal
    src/metal/shaders/ORBDescriptor.metal
    src/metal/shaders/HarrisResponse.metal
)

set(AIR_FILES "")

foreach(shader ${METAL_SHADERS})
    get_filename_component(shader_name ${shader} NAME_WE)
    set(air_file ${CMAKE_BINARY_DIR}/${shader_name}.air)
    
    add_custom_command(
        OUTPUT ${air_file}
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/${shader} -o ${air_file}
        DEPENDS ${CMAKE_SOURCE_DIR}/${shader}
    )
    list(APPEND AIR_FILES ${air_file})
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders.metallib
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${CMAKE_BINARY_DIR}/shaders.metallib
    DEPENDS ${AIR_FILES}
)

add_custom_target(metal_shaders DEPENDS ${CMAKE_BINARY_DIR}/shaders.metallib)

# Shared Sources and Libraries
set(VIO_SOURCES
    src/dataset/EurocLoader.cpp
    src/dataset/TrajectoryWriter.cpp
    src/metal/MetalContext.mm
    src/metal/MetalUndistort.mm
    src/vision/FeatureDetector.cpp
    src/vision/StereoMatcher.cpp
    src/vision/TemporalTracker.mm
    src/vision/FeatureManager.cpp
    src/vision/LoopDetector.cpp
    src/imu/ImuPreintegrator.cpp
    src/optimization/VioOptimizer.cpp
    src/optimization/Factors.cpp
    src/optimization/Marginalization.cpp
    src/core/KeyframePolicy.cpp
    src/core/Profiler.cpp
    src/metal/FastDetect.mm
    src/metal/KLTTracker.mm
    src/metal/HarrisResponse.mm
    src/metal/ORBDescriptor.mm
    src/metal/StereoMatcher.mm
)

set(VIO_LIBS
    Eigen3::Eigen
    ceres
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    ${Pangolin_LIBRARIES}   # --- FIX: Added Pangolin here so main executables can use it ---
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    ${VISION_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Enable ARC for Objective-C++
set(ARC_FILES
    src/main.mm
    src/metal_main.mm
    src/metal/MetalContext.mm
    src/metal/MetalUndistort.mm
    src/vision/TemporalTracker.mm
    src/metal/FastDetect.mm
    src/metal/KLTTracker.mm
    src/metal/HarrisResponse.mm
    src/metal/ORBDescriptor.mm
    src/metal/StereoMatcher.mm
)
set_source_files_properties(${ARC_FILES} PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# CPU Executable
add_executable(vio-metal src/main.mm ${VIO_SOURCES})
add_dependencies(vio-metal metal_shaders)
target_include_directories(vio-metal PRIVATE src/ ${EIGEN3_INCLUDE_DIR})
target_link_libraries(vio-metal PRIVATE ${VIO_LIBS})
target_compile_definitions(vio-metal PRIVATE EIGEN_USE_BLAS)

# GPU Executable
add_executable(vio-metal-gpu src/metal_main.mm ${VIO_SOURCES})
add_dependencies(vio-metal-gpu metal_shaders)
target_include_directories(vio-metal-gpu PRIVATE src/ ${EIGEN3_INCLUDE_DIR})
target_link_libraries(vio-metal-gpu PRIVATE ${VIO_LIBS})
target_compile_definitions(vio-metal-gpu PRIVATE EIGEN_USE_BLAS)

# Tests
add_executable(test_euroc_loader
    test/test_euroc_loader.cpp
    src/dataset/EurocLoader.cpp
)
target_include_directories(test_euroc_loader PRIVATE src/ ${EIGEN3_INCLUDE_DIR})
target_link_libraries(test_euroc_loader PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    ${Pangolin_LIBRARIES}
)