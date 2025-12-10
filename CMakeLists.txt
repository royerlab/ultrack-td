cmake_minimum_required(VERSION 3.15...3.26)

project(ultrack_td LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations for Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Detect if we're building wheels for distribution
# cibuildwheel sets CIBUILDWHEEL, other CI systems set CI
set(IS_DISTRIBUTION_BUILD OFF)
if(DEFINED ENV{CIBUILDWHEEL} OR DEFINED ENV{CI} OR DEFINED ENV{GITHUB_ACTIONS})
  set(IS_DISTRIBUTION_BUILD ON)
  message(STATUS "Detected distribution build - using portable optimizations")
else()
  message(STATUS "Detected local build - using native optimizations")
endif()

# Compiler-specific optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(
    -O2                    # Good optimization level
    -DNDEBUG              # Disable assertions
 )

  # Use native optimizations only for local builds
  if(NOT IS_DISTRIBUTION_BUILD)
    add_compile_options(
      -march=native        # Optimize for the current CPU architecture
      -ffast-math         # Enable fast math optimizations
    )
    message(STATUS "Enabled -march=native and -ffast-math for local build")
  else()
    message(STATUS "Using portable build settings for distribution")
  endif()

elseif(MSVC)
  add_compile_options(
    /O2                    # Maximum optimization
    /DNDEBUG              # Disable assertions
  )

  if(NOT IS_DISTRIBUTION_BUILD)
    add_compile_options(
      /arch:AVX2          # Enable AVX2 instructions
      /fp:fast            # Fast floating-point model
    )
    message(STATUS "Enabled AVX2 and fast-math for local build")
  else()
    message(STATUS "Using portable build settings for distribution")
  endif()
endif()

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  ultrack_td_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/ultrack_td_ext.cpp
)

# Enable link-time optimization (LTO) for better performance
set_target_properties(ultrack_td_ext PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Install directive for scikit-build-core
install(TARGETS ultrack_td_ext LIBRARY DESTINATION ultrack_td)
