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
include doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/depend.make

# Include the progress variables for this target.
include doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/progress.make

# Include the compile flags for this target's objects.
include doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/flags.make

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/flags.make
doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o: doc/snippets/compile_PartialRedux_sum.cpp
doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/doc/snippets/PartialRedux_sum.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets/compile_PartialRedux_sum.cpp

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets/compile_PartialRedux_sum.cpp > CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.i

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets/compile_PartialRedux_sum.cpp -o CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.s

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.requires:
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.requires

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.provides: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.requires
	$(MAKE) -f doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/build.make doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.provides.build
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.provides

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.provides.build: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o

# Object files for target compile_PartialRedux_sum
compile_PartialRedux_sum_OBJECTS = \
"CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o"

# External object files for target compile_PartialRedux_sum
compile_PartialRedux_sum_EXTERNAL_OBJECTS =

doc/snippets/compile_PartialRedux_sum: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o
doc/snippets/compile_PartialRedux_sum: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/build.make
doc/snippets/compile_PartialRedux_sum: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable compile_PartialRedux_sum"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_PartialRedux_sum.dir/link.txt --verbose=$(VERBOSE)
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && ./compile_PartialRedux_sum >/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets/PartialRedux_sum.out

# Rule to build all files generated by this target.
doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/build: doc/snippets/compile_PartialRedux_sum
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/build

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/requires: doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/compile_PartialRedux_sum.cpp.o.requires
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/requires

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/clean:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_PartialRedux_sum.dir/cmake_clean.cmake
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/clean

doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/doc/snippets /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/snippets/CMakeFiles/compile_PartialRedux_sum.dir/depend

