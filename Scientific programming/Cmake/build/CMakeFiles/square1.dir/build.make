# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/valentin/Documents/RW/Cpp/Cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/Documents/RW/Cpp/Cmake/build

# Include any dependencies generated for this target.
include CMakeFiles/square1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/square1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/square1.dir/flags.make

CMakeFiles/square1.dir/main.cpp.o: CMakeFiles/square1.dir/flags.make
CMakeFiles/square1.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/Cpp/Cmake/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/square1.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/square1.dir/main.cpp.o -c /home/valentin/Documents/RW/Cpp/Cmake/main.cpp

CMakeFiles/square1.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/square1.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/Cpp/Cmake/main.cpp > CMakeFiles/square1.dir/main.cpp.i

CMakeFiles/square1.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/square1.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/Cpp/Cmake/main.cpp -o CMakeFiles/square1.dir/main.cpp.s

CMakeFiles/square1.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/square1.dir/main.cpp.o.requires

CMakeFiles/square1.dir/main.cpp.o.provides: CMakeFiles/square1.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/square1.dir/build.make CMakeFiles/square1.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/square1.dir/main.cpp.o.provides

CMakeFiles/square1.dir/main.cpp.o.provides.build: CMakeFiles/square1.dir/main.cpp.o

CMakeFiles/square1.dir/square.cpp.o: CMakeFiles/square1.dir/flags.make
CMakeFiles/square1.dir/square.cpp.o: ../square.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/Cpp/Cmake/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/square1.dir/square.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/square1.dir/square.cpp.o -c /home/valentin/Documents/RW/Cpp/Cmake/square.cpp

CMakeFiles/square1.dir/square.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/square1.dir/square.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/Cpp/Cmake/square.cpp > CMakeFiles/square1.dir/square.cpp.i

CMakeFiles/square1.dir/square.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/square1.dir/square.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/Cpp/Cmake/square.cpp -o CMakeFiles/square1.dir/square.cpp.s

CMakeFiles/square1.dir/square.cpp.o.requires:
.PHONY : CMakeFiles/square1.dir/square.cpp.o.requires

CMakeFiles/square1.dir/square.cpp.o.provides: CMakeFiles/square1.dir/square.cpp.o.requires
	$(MAKE) -f CMakeFiles/square1.dir/build.make CMakeFiles/square1.dir/square.cpp.o.provides.build
.PHONY : CMakeFiles/square1.dir/square.cpp.o.provides

CMakeFiles/square1.dir/square.cpp.o.provides.build: CMakeFiles/square1.dir/square.cpp.o

# Object files for target square1
square1_OBJECTS = \
"CMakeFiles/square1.dir/main.cpp.o" \
"CMakeFiles/square1.dir/square.cpp.o"

# External object files for target square1
square1_EXTERNAL_OBJECTS =

square1: CMakeFiles/square1.dir/main.cpp.o
square1: CMakeFiles/square1.dir/square.cpp.o
square1: CMakeFiles/square1.dir/build.make
square1: CMakeFiles/square1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable square1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/square1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/square1.dir/build: square1
.PHONY : CMakeFiles/square1.dir/build

CMakeFiles/square1.dir/requires: CMakeFiles/square1.dir/main.cpp.o.requires
CMakeFiles/square1.dir/requires: CMakeFiles/square1.dir/square.cpp.o.requires
.PHONY : CMakeFiles/square1.dir/requires

CMakeFiles/square1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/square1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/square1.dir/clean

CMakeFiles/square1.dir/depend:
	cd /home/valentin/Documents/RW/Cpp/Cmake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Documents/RW/Cpp/Cmake /home/valentin/Documents/RW/Cpp/Cmake /home/valentin/Documents/RW/Cpp/Cmake/build /home/valentin/Documents/RW/Cpp/Cmake/build /home/valentin/Documents/RW/Cpp/Cmake/build/CMakeFiles/square1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/square1.dir/depend

