cmake_minimum_required(VERSION 3.10)
project(ExtKalmanSlam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Find required packages
find_package(Eigen3 REQUIRED)

# Header-only library. include/robot_tools/ is kept identical to
# slam3d_tools/include/robot_tools/ (see scripts/check_sync.sh).
add_library(robot_tools INTERFACE)
target_include_directories(robot_tools INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(robot_tools INTERFACE Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")

# Add main executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} robot_tools)

# ============================================================================
# Testing
# ============================================================================
option(BUILD_TESTS "Build the tests" ON)

if(BUILD_TESTS)
  # Find Google Test
  find_package(GTest REQUIRED)
  include(GoogleTest)

  # Enable testing
  enable_testing()

  # Add test executable
  add_executable(test_ext_kalman_slam test_ext_kalman_slam.cpp)

  # Link test libraries
  target_link_libraries(test_ext_kalman_slam
    robot_tools
    GTest::GTest
    GTest::Main
  )

  # Discover tests
  gtest_discover_tests(test_ext_kalman_slam)
endif()
