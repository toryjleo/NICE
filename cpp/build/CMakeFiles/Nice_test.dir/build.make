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
CMAKE_SOURCE_DIR = /home/jason.b/Desktop/Github/NICE/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason.b/Desktop/Github/NICE/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/Nice_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Nice_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Nice_test.dir/flags.make

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o: CMakeFiles/Nice_test.dir/flags.make
CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o: ../test/cpu_operations_test/transpose_test.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o -c /home/jason.b/Desktop/Github/NICE/cpp/test/cpu_operations_test/transpose_test.cc

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/test/cpu_operations_test/transpose_test.cc > CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.i

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/test/cpu_operations_test/transpose_test.cc -o CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.s

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.requires:
.PHONY : CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.requires

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.provides: CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.requires
	$(MAKE) -f CMakeFiles/Nice_test.dir/build.make CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.provides.build
.PHONY : CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.provides

CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.provides.build: CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o: CMakeFiles/Nice_test.dir/flags.make
CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o: ../test/util_test/from_file_test.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o -c /home/jason.b/Desktop/Github/NICE/cpp/test/util_test/from_file_test.cc

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/test/util_test/from_file_test.cc > CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.i

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/test/util_test/from_file_test.cc -o CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.s

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.requires:
.PHONY : CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.requires

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.provides: CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.requires
	$(MAKE) -f CMakeFiles/Nice_test.dir/build.make CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.provides.build
.PHONY : CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.provides

CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.provides.build: CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o

# Object files for target Nice_test
Nice_test_OBJECTS = \
"CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o" \
"CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o"

# External object files for target Nice_test
Nice_test_EXTERNAL_OBJECTS =

Nice_test: CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o
Nice_test: CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o
Nice_test: CMakeFiles/Nice_test.dir/build.make
Nice_test: gtest/src/googletest-build/googlemock/gtest/libgtest.a
Nice_test: gtest/src/googletest-build/googlemock/gtest/libgtest_main.a
Nice_test: libNice.so
Nice_test: CMakeFiles/Nice_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Nice_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Nice_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Nice_test.dir/build: Nice_test
.PHONY : CMakeFiles/Nice_test.dir/build

CMakeFiles/Nice_test.dir/requires: CMakeFiles/Nice_test.dir/test/cpu_operations_test/transpose_test.cc.o.requires
CMakeFiles/Nice_test.dir/requires: CMakeFiles/Nice_test.dir/test/util_test/from_file_test.cc.o.requires
.PHONY : CMakeFiles/Nice_test.dir/requires

CMakeFiles/Nice_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Nice_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Nice_test.dir/clean

CMakeFiles/Nice_test.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp /home/jason.b/Desktop/Github/NICE/cpp /home/jason.b/Desktop/Github/NICE/cpp/build /home/jason.b/Desktop/Github/NICE/cpp/build /home/jason.b/Desktop/Github/NICE/cpp/build/CMakeFiles/Nice_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Nice_test.dir/depend

