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
include CMakeFiles/gramschmidt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gramschmidt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gramschmidt.dir/flags.make

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o: CMakeFiles/gramschmidt.dir/flags.make
CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o: gramschmidt.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/valentin/Documents/RW/CSE/PS0/solutions/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o -c /home/valentin/Documents/RW/CSE/PS0/solutions/gramschmidt.cpp

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gramschmidt.dir/gramschmidt.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/valentin/Documents/RW/CSE/PS0/solutions/gramschmidt.cpp > CMakeFiles/gramschmidt.dir/gramschmidt.cpp.i

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gramschmidt.dir/gramschmidt.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/valentin/Documents/RW/CSE/PS0/solutions/gramschmidt.cpp -o CMakeFiles/gramschmidt.dir/gramschmidt.cpp.s

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.requires:
.PHONY : CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.requires

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.provides: CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.requires
	$(MAKE) -f CMakeFiles/gramschmidt.dir/build.make CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.provides.build
.PHONY : CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.provides

CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.provides.build: CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o

# Object files for target gramschmidt
gramschmidt_OBJECTS = \
"CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o"

# External object files for target gramschmidt
gramschmidt_EXTERNAL_OBJECTS =

gramschmidt: CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o
gramschmidt: CMakeFiles/gramschmidt.dir/build.make
gramschmidt: CMakeFiles/gramschmidt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable gramschmidt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gramschmidt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gramschmidt.dir/build: gramschmidt
.PHONY : CMakeFiles/gramschmidt.dir/build

CMakeFiles/gramschmidt.dir/requires: CMakeFiles/gramschmidt.dir/gramschmidt.cpp.o.requires
.PHONY : CMakeFiles/gramschmidt.dir/requires

CMakeFiles/gramschmidt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gramschmidt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gramschmidt.dir/clean

CMakeFiles/gramschmidt.dir/depend:
	cd /home/valentin/Documents/RW/CSE/PS0/solutions && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions /home/valentin/Documents/RW/CSE/PS0/solutions/CMakeFiles/gramschmidt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gramschmidt.dir/depend

