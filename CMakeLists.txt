cmake_minimum_required(VERSION 3.10)
project(ExtKalmanSlam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Find required packages
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Add main executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARIES})

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
    ${EIGEN3_LIBRARIES}
    GTest::GTest
    GTest::Main
  )

  # Discover tests
  gtest_discover_tests(test_ext_kalman_slam)
endif()
