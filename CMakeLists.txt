cmake_minimum_required(VERSION 3.2)

SET(TARGET test)
project(${TARGET} C CXX)

include(~/Desktop/ChiTech/chi-tech/CHI_RESOURCES/Macros/Downstream.cmake)
include(~/Desktop/ChiTech/chi-tech/CHI_RESOURCES/Macros/Filter.cmake)

set(SOURCES "test3b.cc")
file (GLOB_RECURSE MORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Solver/*.cc")
set(SOURCES ${SOURCES} ${MORE_SOURCES})

set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 ")

add_executable(${TARGET} "${SOURCES}")
target_link_libraries(${TARGET} ${CHI_LIBS})

