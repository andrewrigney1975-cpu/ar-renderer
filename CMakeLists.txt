cmake_minimum_required(VERSION 3.24)
project(prender VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PR_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /MP /wd4100 /wd4324 /wd4127)
  add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi>
                      $<$<CONFIG:Release>:/GL> /arch:AVX2)
  add_link_options($<$<CONFIG:Release>:/LTCG>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_library(pr_core STATIC
  src/core/math.cpp
  src/core/sampling.cpp
  src/core/spectrum.cpp
  src/core/color.cpp
  src/core/rgb2spec.cpp
  src/core/parallel.cpp
  src/core/log.cpp
  src/core/image.cpp
  src/core/film.cpp
  src/geometry/shape.cpp
  src/geometry/mesh_io.cpp
  src/geometry/bvh.cpp
  src/materials/bxdf.cpp
  src/materials/texture.cpp
  src/materials/material.cpp
  src/media/medium.cpp
  src/lights/light.cpp
  src/cameras/camera.cpp
  src/scene/scene.cpp
  src/scene/scene_loader.cpp
  src/integrators/integrator.cpp
  src/integrators/path.cpp
  src/integrators/bdpt.cpp
  src/integrators/mlt.cpp
)
target_include_directories(pr_core PUBLIC src third_party)

add_executable(prender src/cli/main.cpp)
target_link_libraries(prender PRIVATE pr_core)

if(PR_BUILD_TESTS)
  enable_testing()
  add_executable(prender_tests
    tests/test_main.cpp
    tests/test_spectrum.cpp
    tests/test_bsdf.cpp
    tests/test_render.cpp
  )
  target_link_libraries(prender_tests PRIVATE pr_core)
  target_compile_definitions(prender_tests PRIVATE PR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME prender_tests COMMAND prender_tests)
endif()
