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
include CMakeFiles/quadsingint.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/quadsingint.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/quadsingint.dir/flags.make

CMakeFiles/quadsingint.dir/quadsingint.cpp.o: CMakeFiles/quadsingint.dir/flags.make
CMakeFiles/quadsingint.dir/quadsingint.cpp.o: quadsingint.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/quadsingint.dir/quadsingint.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/quadsingint.dir/quadsingint.cpp.o -c /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadsingint.cpp

CMakeFiles/quadsingint.dir/quadsingint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quadsingint.dir/quadsingint.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadsingint.cpp > CMakeFiles/quadsingint.dir/quadsingint.cpp.i

CMakeFiles/quadsingint.dir/quadsingint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quadsingint.dir/quadsingint.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/quadsingint.cpp -o CMakeFiles/quadsingint.dir/quadsingint.cpp.s

CMakeFiles/quadsingint.dir/quadsingint.cpp.o.requires:
.PHONY : CMakeFiles/quadsingint.dir/quadsingint.cpp.o.requires

CMakeFiles/quadsingint.dir/quadsingint.cpp.o.provides: CMakeFiles/quadsingint.dir/quadsingint.cpp.o.requires
	$(MAKE) -f CMakeFiles/quadsingint.dir/build.make CMakeFiles/quadsingint.dir/quadsingint.cpp.o.provides.build
.PHONY : CMakeFiles/quadsingint.dir/quadsingint.cpp.o.provides

CMakeFiles/quadsingint.dir/quadsingint.cpp.o.provides.build: CMakeFiles/quadsingint.dir/quadsingint.cpp.o

CMakeFiles/quadsingint.dir/gauleg.cpp.o: CMakeFiles/quadsingint.dir/flags.make
CMakeFiles/quadsingint.dir/gauleg.cpp.o: gauleg.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/quadsingint.dir/gauleg.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/quadsingint.dir/gauleg.cpp.o -c /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/gauleg.cpp

CMakeFiles/quadsingint.dir/gauleg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quadsingint.dir/gauleg.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/gauleg.cpp > CMakeFiles/quadsingint.dir/gauleg.cpp.i

CMakeFiles/quadsingint.dir/gauleg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quadsingint.dir/gauleg.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/gauleg.cpp -o CMakeFiles/quadsingint.dir/gauleg.cpp.s

CMakeFiles/quadsingint.dir/gauleg.cpp.o.requires:
.PHONY : CMakeFiles/quadsingint.dir/gauleg.cpp.o.requires

CMakeFiles/quadsingint.dir/gauleg.cpp.o.provides: CMakeFiles/quadsingint.dir/gauleg.cpp.o.requires
	$(MAKE) -f CMakeFiles/quadsingint.dir/build.make CMakeFiles/quadsingint.dir/gauleg.cpp.o.provides.build
.PHONY : CMakeFiles/quadsingint.dir/gauleg.cpp.o.provides

CMakeFiles/quadsingint.dir/gauleg.cpp.o.provides.build: CMakeFiles/quadsingint.dir/gauleg.cpp.o

# Object files for target quadsingint
quadsingint_OBJECTS = \
"CMakeFiles/quadsingint.dir/quadsingint.cpp.o" \
"CMakeFiles/quadsingint.dir/gauleg.cpp.o"

# External object files for target quadsingint
quadsingint_EXTERNAL_OBJECTS =

quadsingint: CMakeFiles/quadsingint.dir/quadsingint.cpp.o
quadsingint: CMakeFiles/quadsingint.dir/gauleg.cpp.o
quadsingint: CMakeFiles/quadsingint.dir/build.make
quadsingint: CMakeFiles/quadsingint.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable quadsingint"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quadsingint.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/quadsingint.dir/build: quadsingint
.PHONY : CMakeFiles/quadsingint.dir/build

CMakeFiles/quadsingint.dir/requires: CMakeFiles/quadsingint.dir/quadsingint.cpp.o.requires
CMakeFiles/quadsingint.dir/requires: CMakeFiles/quadsingint.dir/gauleg.cpp.o.requires
.PHONY : CMakeFiles/quadsingint.dir/requires

CMakeFiles/quadsingint.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/quadsingint.dir/cmake_clean.cmake
.PHONY : CMakeFiles/quadsingint.dir/clean

CMakeFiles/quadsingint.dir/depend:
	cd /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11 /home/valentin/Documents/RW/CSE/PS11/solutions_ps11/CMakeFiles/quadsingint.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/quadsingint.dir/depend
