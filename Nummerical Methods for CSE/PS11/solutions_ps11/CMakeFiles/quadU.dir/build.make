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
CMAKE_SOURCE_DIR = /home/valentin/Documents/RW/CSE/PS11/solutions_ps11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/Documents/RW/CSE/PS11/solutions_ps11

# Include any dependencies generated for this target.
include CMakeFiles/quadU.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/quadU.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/quadU.dir/flags.make

CMakeFiles/quadU.dir/quadU.cpp.o: CMakeFiles/quadU.dir/flags.make
CMakeFiles/quadU.dir/quadU.cpp.o: quadU.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/quadU.dir/quadU.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/quadU.dir/quadU.cpp.o -c /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadU.cpp

CMakeFiles/quadU.dir/quadU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quadU.dir/quadU.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadU.cpp > CMakeFiles/quadU.dir/quadU.cpp.i

CMakeFiles/quadU.dir/quadU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quadU.dir/quadU.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadU.cpp -o CMakeFiles/quadU.dir/quadU.cpp.s

CMakeFiles/quadU.dir/quadU.cpp.o.requires:
.PHONY : CMakeFiles/quadU.dir/quadU.cpp.o.requires

CMakeFiles/quadU.dir/quadU.cpp.o.provides: CMakeFiles/quadU.dir/quadU.cpp.o.requires
	$(MAKE) -f CMakeFiles/quadU.dir/build.make CMakeFiles/quadU.dir/quadU.cpp.o.provides.build
.PHONY : CMakeFiles/quadU.dir/quadU.cpp.o.provides

CMakeFiles/quadU.dir/quadU.cpp.o.provides.build: CMakeFiles/quadU.dir/quadU.cpp.o

# Object files for target quadU
quadU_OBJECTS = \
"CMakeFiles/quadU.dir/quadU.cpp.o"

# External object files for target quadU
quadU_EXTERNAL_OBJECTS =

quadU: CMakeFiles/quadU.dir/quadU.cpp.o
quadU: CMakeFiles/quadU.dir/build.make
quadU: CMakeFiles/quadU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable quadU"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quadU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/quadU.dir/build: quadU
.PHONY : CMakeFiles/quadU.dir/build

CMakeFiles/quadU.dir/requires: CMakeFiles/quadU.dir/quadU.cpp.o.requires
.PHONY : CMakeFiles/quadU.dir/requires

CMakeFiles/quadU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/quadU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/quadU.dir/clean

CMakeFiles/quadU.dir/depend:
	cd /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/CMakeFiles/quadU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/quadU.dir/depend
