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
include unsupported/test/CMakeFiles/autodiff_1.dir/depend.make

# Include the progress variables for this target.
include unsupported/test/CMakeFiles/autodiff_1.dir/progress.make

# Include the compile flags for this target's objects.
include unsupported/test/CMakeFiles/autodiff_1.dir/flags.make

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o: unsupported/test/CMakeFiles/autodiff_1.dir/flags.make
unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/test/autodiff.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/autodiff_1.dir/autodiff.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/test/autodiff.cpp

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/autodiff_1.dir/autodiff.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/test/autodiff.cpp > CMakeFiles/autodiff_1.dir/autodiff.cpp.i

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/autodiff_1.dir/autodiff.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/test/autodiff.cpp -o CMakeFiles/autodiff_1.dir/autodiff.cpp.s

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.requires:
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.requires

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.provides: unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.requires
	$(MAKE) -f unsupported/test/CMakeFiles/autodiff_1.dir/build.make unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.provides.build
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.provides

unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.provides.build: unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o

# Object files for target autodiff_1
autodiff_1_OBJECTS = \
"CMakeFiles/autodiff_1.dir/autodiff.cpp.o"

# External object files for target autodiff_1
autodiff_1_EXTERNAL_OBJECTS =

unsupported/test/autodiff_1: unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o
unsupported/test/autodiff_1: unsupported/test/CMakeFiles/autodiff_1.dir/build.make
unsupported/test/autodiff_1: unsupported/test/CMakeFiles/autodiff_1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable autodiff_1"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/autodiff_1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unsupported/test/CMakeFiles/autodiff_1.dir/build: unsupported/test/autodiff_1
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/build

unsupported/test/CMakeFiles/autodiff_1.dir/requires: unsupported/test/CMakeFiles/autodiff_1.dir/autodiff.cpp.o.requires
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/requires

unsupported/test/CMakeFiles/autodiff_1.dir/clean:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test && $(CMAKE_COMMAND) -P CMakeFiles/autodiff_1.dir/cmake_clean.cmake
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/clean

unsupported/test/CMakeFiles/autodiff_1.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/unsupported/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/unsupported/test/CMakeFiles/autodiff_1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unsupported/test/CMakeFiles/autodiff_1.dir/depend

