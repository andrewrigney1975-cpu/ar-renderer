cmake_minimum_required(VERSION 3.24)
project(prender VERSION 0.3.0 LANGUAGES C CXX)

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
  add_compile_options(/utf-8 /MP)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/Zc:__cplusplus;/wd4100;/wd4324;/wd4127>")
  add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi>
                      $<$<CONFIG:Release>:/GL> /arch:AVX2)
  add_link_options($<$<CONFIG:Release>:/LTCG>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_library(pr_core STATIC
  src/core/math.cpp
  src/core/noise.cpp
  src/core/sampling.cpp
  src/core/spectrum.cpp
  src/core/color.cpp
  src/core/rgb2spec.cpp
  src/core/parallel.cpp
  src/core/log.cpp
  src/core/image.cpp
  src/core/exr_read.cpp
  third_party/tinyexr/miniz.c
  src/core/film.cpp
  src/geometry/shape.cpp
  src/geometry/mesh_io.cpp
  src/geometry/bvh.cpp
  src/materials/bxdf.cpp
  src/materials/texture.cpp
  src/materials/material.cpp
  src/media/medium.cpp
  src/media/nanovdb_field.cpp
  src/lights/light.cpp
  src/lights/ies.cpp
  src/lights/sky.cpp
  src/lights/light_bvh.cpp
  src/cameras/camera.cpp
  src/cameras/realistic.cpp
  src/scene/scene.cpp
  src/scene/scene_loader.cpp
  src/scene/import_gltf.cpp
  src/scene/import_pbrt.cpp
  src/integrators/integrator.cpp
  src/integrators/aov.cpp
  src/integrators/gpu_path.cpp
  src/integrators/path.cpp
  src/integrators/bdpt.cpp
  src/integrators/mlt.cpp
)
target_include_directories(pr_core PUBLIC src third_party third_party/nanovdb third_party/tinyexr)

# Third-party noise.
set_source_files_properties(third_party/tinyexr/miniz.c PROPERTIES COMPILE_OPTIONS "$<$<C_COMPILER_ID:MSVC>:/W0>")
set_source_files_properties(src/media/nanovdb_field.cpp PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/wd4244;/wd4267;/wd4146>")

add_executable(prender src/cli/main.cpp)
target_link_libraries(prender PRIVATE pr_core)

if(PR_BUILD_TESTS)
  enable_testing()
  add_executable(prender_tests
    tests/test_main.cpp
    tests/test_spectrum.cpp
    tests/test_bsdf.cpp
    tests/test_render.cpp
    tests/test_media.cpp
    tests/test_lights.cpp
    tests/test_import.cpp
    tests/test_checkpoint.cpp
    tests/test_camera.cpp
  )
  target_link_libraries(prender_tests PRIVATE pr_core)
  if(MSVC)
    target_compile_options(prender_tests PRIVATE /wd5285)  # doctest specializes std::tuple
  endif()
  target_compile_definitions(prender_tests PRIVATE PR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME prender_tests COMMAND prender_tests)
endif()
