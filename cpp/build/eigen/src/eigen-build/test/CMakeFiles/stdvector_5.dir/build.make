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
include test/CMakeFiles/stdvector_5.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/stdvector_5.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/stdvector_5.dir/flags.make

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o: test/CMakeFiles/stdvector_5.dir/flags.make
test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/stdvector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stdvector_5.dir/stdvector.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/stdvector.cpp

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stdvector_5.dir/stdvector.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/stdvector.cpp > CMakeFiles/stdvector_5.dir/stdvector.cpp.i

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stdvector_5.dir/stdvector.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test/stdvector.cpp -o CMakeFiles/stdvector_5.dir/stdvector.cpp.s

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.requires:
.PHONY : test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.requires

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.provides: test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/stdvector_5.dir/build.make test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.provides.build
.PHONY : test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.provides

test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.provides.build: test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o

# Object files for target stdvector_5
stdvector_5_OBJECTS = \
"CMakeFiles/stdvector_5.dir/stdvector.cpp.o"

# External object files for target stdvector_5
stdvector_5_EXTERNAL_OBJECTS =

test/stdvector_5: test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o
test/stdvector_5: test/CMakeFiles/stdvector_5.dir/build.make
test/stdvector_5: test/CMakeFiles/stdvector_5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable stdvector_5"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stdvector_5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/stdvector_5.dir/build: test/stdvector_5
.PHONY : test/CMakeFiles/stdvector_5.dir/build

test/CMakeFiles/stdvector_5.dir/requires: test/CMakeFiles/stdvector_5.dir/stdvector.cpp.o.requires
.PHONY : test/CMakeFiles/stdvector_5.dir/requires

test/CMakeFiles/stdvector_5.dir/clean:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/stdvector_5.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/stdvector_5.dir/clean

test/CMakeFiles/stdvector_5.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/test/CMakeFiles/stdvector_5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/stdvector_5.dir/depend

