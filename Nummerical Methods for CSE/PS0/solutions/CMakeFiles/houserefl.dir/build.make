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
CMAKE_SOURCE_DIR = /home/valentin/Documents/RW/CSE/PS0/solutions

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/Documents/RW/CSE/PS0/solutions

# Include any dependencies generated for this target.
include CMakeFiles/houserefl.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/houserefl.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/houserefl.dir/flags.make

CMakeFiles/houserefl.dir/houserefl.cpp.o: CMakeFiles/houserefl.dir/flags.make
CMakeFiles/houserefl.dir/houserefl.cpp.o: houserefl.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/CSE/PS0/solutions/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/houserefl.dir/houserefl.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/houserefl.dir/houserefl.cpp.o -c /home/valentin/Documents/RW/CSE/PS0/solutions/houserefl.cpp

CMakeFiles/houserefl.dir/houserefl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/houserefl.dir/houserefl.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/CSE/PS0/solutions/houserefl.cpp > CMakeFiles/houserefl.dir/houserefl.cpp.i

CMakeFiles/houserefl.dir/houserefl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/houserefl.dir/houserefl.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/CSE/PS0/solutions/houserefl.cpp -o CMakeFiles/houserefl.dir/houserefl.cpp.s

CMakeFiles/houserefl.dir/houserefl.cpp.o.requires:
.PHONY : CMakeFiles/houserefl.dir/houserefl.cpp.o.requires

CMakeFiles/houserefl.dir/houserefl.cpp.o.provides: CMakeFiles/houserefl.dir/houserefl.cpp.o.requires
	$(MAKE) -f CMakeFiles/houserefl.dir/build.make CMakeFiles/houserefl.dir/houserefl.cpp.o.provides.build
.PHONY : CMakeFiles/houserefl.dir/houserefl.cpp.o.provides

CMakeFiles/houserefl.dir/houserefl.cpp.o.provides.build: CMakeFiles/houserefl.dir/houserefl.cpp.o

# Object files for target houserefl
houserefl_OBJECTS = \
"CMakeFiles/houserefl.dir/houserefl.cpp.o"

# External object files for target houserefl
houserefl_EXTERNAL_OBJECTS =

houserefl: CMakeFiles/houserefl.dir/houserefl.cpp.o
houserefl: CMakeFiles/houserefl.dir/build.make
houserefl: CMakeFiles/houserefl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable houserefl"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/houserefl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/houserefl.dir/build: houserefl
.PHONY : CMakeFiles/houserefl.dir/build

CMakeFiles/houserefl.dir/requires: CMakeFiles/houserefl.dir/houserefl.cpp.o.requires
.PHONY : CMakeFiles/houserefl.dir/requires

CMakeFiles/houserefl.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/houserefl.dir/cmake_clean.cmake
.PHONY : CMakeFiles/houserefl.dir/clean

CMakeFiles/houserefl.dir/depend:
	cd /home/valentin/Documents/RW/CSE/PS0/solutions && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions/CMakeFiles/houserefl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/houserefl.dir/depend

