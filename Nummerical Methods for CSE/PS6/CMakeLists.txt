cmake_minimum_required(VERSION 2.8)

project(CSE_PS6_Valentin)

set(CMAKE_CXX_FLAGS "-std=c++0x")

add_library(timer STATIC timer.cpp)

add_executable(horner horner.cpp)

target_link_libraries(horner timer)

install(TARGETS timer DESTINATION lib)

install(FILES timer.hpp DESTINATION include)

install(TARGETS horner DESTINATION bin)
