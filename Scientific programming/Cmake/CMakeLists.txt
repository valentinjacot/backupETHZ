cmake_minimum_required(VERSION 2.8)
project(Square)
# Just compile the square1 program from source files
add_executable(square1 main.cpp square.cpp)
# install the programs into the bin subdir
install(TARGETS square1 DESTINATION bin)

