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
include blas/CMakeFiles/eigen_blas_static.dir/depend.make

# Include the progress variables for this target.
include blas/CMakeFiles/eigen_blas_static.dir/progress.make

# Include the compile flags for this target's objects.
include blas/CMakeFiles/eigen_blas_static.dir/flags.make

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o: blas/CMakeFiles/eigen_blas_static.dir/flags.make
blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/single.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eigen_blas_static.dir/single.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/single.cpp

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eigen_blas_static.dir/single.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/single.cpp > CMakeFiles/eigen_blas_static.dir/single.cpp.i

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eigen_blas_static.dir/single.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/single.cpp -o CMakeFiles/eigen_blas_static.dir/single.cpp.s

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.requires:
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.requires

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.provides: blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.requires
	$(MAKE) -f blas/CMakeFiles/eigen_blas_static.dir/build.make blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.provides.build
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.provides

blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.provides.build: blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o: blas/CMakeFiles/eigen_blas_static.dir/flags.make
blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/double.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eigen_blas_static.dir/double.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/double.cpp

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eigen_blas_static.dir/double.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/double.cpp > CMakeFiles/eigen_blas_static.dir/double.cpp.i

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eigen_blas_static.dir/double.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/double.cpp -o CMakeFiles/eigen_blas_static.dir/double.cpp.s

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.requires:
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.requires

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.provides: blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.requires
	$(MAKE) -f blas/CMakeFiles/eigen_blas_static.dir/build.make blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.provides.build
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.provides

blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.provides.build: blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o: blas/CMakeFiles/eigen_blas_static.dir/flags.make
blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_single.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_single.cpp

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eigen_blas_static.dir/complex_single.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_single.cpp > CMakeFiles/eigen_blas_static.dir/complex_single.cpp.i

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eigen_blas_static.dir/complex_single.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_single.cpp -o CMakeFiles/eigen_blas_static.dir/complex_single.cpp.s

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.requires:
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.requires

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.provides: blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.requires
	$(MAKE) -f blas/CMakeFiles/eigen_blas_static.dir/build.make blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.provides.build
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.provides

blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.provides.build: blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o: blas/CMakeFiles/eigen_blas_static.dir/flags.make
blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_double.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_double.cpp

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eigen_blas_static.dir/complex_double.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_double.cpp > CMakeFiles/eigen_blas_static.dir/complex_double.cpp.i

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eigen_blas_static.dir/complex_double.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/complex_double.cpp -o CMakeFiles/eigen_blas_static.dir/complex_double.cpp.s

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.requires:
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.requires

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.provides: blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.requires
	$(MAKE) -f blas/CMakeFiles/eigen_blas_static.dir/build.make blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.provides.build
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.provides

blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.provides.build: blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o: blas/CMakeFiles/eigen_blas_static.dir/flags.make
blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o: /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/xerbla.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o -c /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/xerbla.cpp

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eigen_blas_static.dir/xerbla.cpp.i"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/xerbla.cpp > CMakeFiles/eigen_blas_static.dir/xerbla.cpp.i

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eigen_blas_static.dir/xerbla.cpp.s"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas/xerbla.cpp -o CMakeFiles/eigen_blas_static.dir/xerbla.cpp.s

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.requires:
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.requires

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.provides: blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.requires
	$(MAKE) -f blas/CMakeFiles/eigen_blas_static.dir/build.make blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.provides.build
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.provides

blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.provides.build: blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o

# Object files for target eigen_blas_static
eigen_blas_static_OBJECTS = \
"CMakeFiles/eigen_blas_static.dir/single.cpp.o" \
"CMakeFiles/eigen_blas_static.dir/double.cpp.o" \
"CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o" \
"CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o" \
"CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o"

# External object files for target eigen_blas_static
eigen_blas_static_EXTERNAL_OBJECTS =

blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/build.make
blas/libeigen_blas_static.a: blas/CMakeFiles/eigen_blas_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libeigen_blas_static.a"
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && $(CMAKE_COMMAND) -P CMakeFiles/eigen_blas_static.dir/cmake_clean_target.cmake
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/eigen_blas_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
blas/CMakeFiles/eigen_blas_static.dir/build: blas/libeigen_blas_static.a
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/build

blas/CMakeFiles/eigen_blas_static.dir/requires: blas/CMakeFiles/eigen_blas_static.dir/single.cpp.o.requires
blas/CMakeFiles/eigen_blas_static.dir/requires: blas/CMakeFiles/eigen_blas_static.dir/double.cpp.o.requires
blas/CMakeFiles/eigen_blas_static.dir/requires: blas/CMakeFiles/eigen_blas_static.dir/complex_single.cpp.o.requires
blas/CMakeFiles/eigen_blas_static.dir/requires: blas/CMakeFiles/eigen_blas_static.dir/complex_double.cpp.o.requires
blas/CMakeFiles/eigen_blas_static.dir/requires: blas/CMakeFiles/eigen_blas_static.dir/xerbla.cpp.o.requires
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/requires

blas/CMakeFiles/eigen_blas_static.dir/clean:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas && $(CMAKE_COMMAND) -P CMakeFiles/eigen_blas_static.dir/cmake_clean.cmake
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/clean

blas/CMakeFiles/eigen_blas_static.dir/depend:
	cd /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen/blas /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas /home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/eigen-build/blas/CMakeFiles/eigen_blas_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : blas/CMakeFiles/eigen_blas_static.dir/depend

