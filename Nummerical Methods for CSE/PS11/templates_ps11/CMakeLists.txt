cmake_minimum_required(VERSION 2.8)

project(PS11_Valentin)

set(CMAKE_CXX_FLAGS "-Wall -I/home/valentin/Eigen -std=c++0x")
# Create library
add_library(gauleg STATIC gauleg.cpp)

# Create executable
add_executable(main quadsingint_template.cpp)
add_executable(sol quadsingint.cpp)
add_executable(laser laserquad_template.cpp)

target_link_libraries(main gauleg)
target_link_libraries(sol gauleg)
target_link_libraries(laser gauleg)

install(FILES gauleg.hpp DESTINATION include)


install(TARGETS gauleg
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# Install executable
install(TARGETS laser main sol DESTINATION bin)
