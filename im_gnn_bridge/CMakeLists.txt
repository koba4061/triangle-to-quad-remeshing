cmake_minimum_required(VERSION 3.15)
project(im_gnn_bridge LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

set(INSTANT_MESHES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../instant_meshes" CACHE PATH "Instant Meshes source")

if(EXISTS "${INSTANT_MESHES_DIR}/src/batch.cpp")
  message(STATUS "Building im_gnn_bridge WITH Instant Meshes: ${INSTANT_MESHES_DIR}")
  set(HAS_IM TRUE)

  if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS __TBB_NO_IMPLICIT_LINKAGE)
    add_compile_options(/MP /EHsc /bigobj /std:c++14)
  else()
    add_compile_options(-fPIC -Wno-changes-meaning -fpermissive)
  endif()

  set(TBB_BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(TBB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
  set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
  set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_subdirectory(${INSTANT_MESHES_DIR}/ext/tbb ${CMAKE_BINARY_DIR}/tbb EXCLUDE_FROM_ALL)

  set(IM_SRC_DIR ${INSTANT_MESHES_DIR}/src)
  set(IM_SOURCES
    ${IM_SRC_DIR}/meshio.cpp
    ${IM_SRC_DIR}/normal.cpp
    ${IM_SRC_DIR}/adjacency.cpp
    ${IM_SRC_DIR}/meshstats.cpp
    ${IM_SRC_DIR}/hierarchy.cpp
    ${IM_SRC_DIR}/extract.cpp
    ${IM_SRC_DIR}/field.cpp
    ${IM_SRC_DIR}/bvh.cpp
    ${IM_SRC_DIR}/subdivide.cpp
    ${IM_SRC_DIR}/reorder.cpp
    ${IM_SRC_DIR}/serializer.cpp
    ${IM_SRC_DIR}/batch.cpp
    ${IM_SRC_DIR}/smoothcurve.cpp
    ${IM_SRC_DIR}/cleanup.cpp
    ${IM_SRC_DIR}/dedge.cpp
    ${INSTANT_MESHES_DIR}/ext/rply/rply.c
  )

  pybind11_add_module(im_gnn_bridge
    bridge.cpp
    batch_gnn.cpp
    ${IM_SOURCES}
  )

  target_compile_definitions(im_gnn_bridge PRIVATE HAS_INSTANT_MESHES=1)
  target_include_directories(im_gnn_bridge PRIVATE
    ${IM_SRC_DIR}
    ${INSTANT_MESHES_DIR}/ext/nanogui/ext/eigen
    ${INSTANT_MESHES_DIR}/ext/tbb/include
    ${INSTANT_MESHES_DIR}/ext/dset
    ${INSTANT_MESHES_DIR}/ext/pss
    ${INSTANT_MESHES_DIR}/ext/pcg32
    ${INSTANT_MESHES_DIR}/ext/rply
    ${INSTANT_MESHES_DIR}/ext/half
  )
  target_link_libraries(im_gnn_bridge PRIVATE tbb_static)

else()
  message(WARNING "Instant Meshes not found at ${INSTANT_MESHES_DIR} — building stub module")
  pybind11_add_module(im_gnn_bridge bridge.cpp)
  target_compile_definitions(im_gnn_bridge PRIVATE HAS_INSTANT_MESHES=0)
endif()
