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
CMAKE_SOURCE_DIR = /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build

# Include any dependencies generated for this target.
include test/CMakeFiles/miscmatrices_4.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/miscmatrices_4.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/miscmatrices_4.dir/flags.make

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o: test/CMakeFiles/miscmatrices_4.dir/flags.make
test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/miscmatrices.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/miscmatrices.cpp

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/miscmatrices.cpp > CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.i

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/miscmatrices.cpp -o CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.s

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.requires:
.PHONY : test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.requires

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.provides: test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/miscmatrices_4.dir/build.make test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.provides.build
.PHONY : test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.provides

test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.provides.build: test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o

# Object files for target miscmatrices_4
miscmatrices_4_OBJECTS = \
"CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o"

# External object files for target miscmatrices_4
miscmatrices_4_EXTERNAL_OBJECTS =

test/miscmatrices_4: test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o
test/miscmatrices_4: test/CMakeFiles/miscmatrices_4.dir/build.make
test/miscmatrices_4: test/CMakeFiles/miscmatrices_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable miscmatrices_4"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/miscmatrices_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/miscmatrices_4.dir/build: test/miscmatrices_4
.PHONY : test/CMakeFiles/miscmatrices_4.dir/build

test/CMakeFiles/miscmatrices_4.dir/requires: test/CMakeFiles/miscmatrices_4.dir/miscmatrices.cpp.o.requires
.PHONY : test/CMakeFiles/miscmatrices_4.dir/requires

test/CMakeFiles/miscmatrices_4.dir/clean:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/miscmatrices_4.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/miscmatrices_4.dir/clean

test/CMakeFiles/miscmatrices_4.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test/CMakeFiles/miscmatrices_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/miscmatrices_4.dir/depend

