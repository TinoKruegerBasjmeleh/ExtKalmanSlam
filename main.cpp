/******************************************************************************
 *  Copyright (c) 2025, KION Group                                            *
 *  All rights reserved.                                                      *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Authors
 *   Tino Krueger-Basjmeleh (tino.krueger@kiongroup.com)
 ******************************************************************************/
#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include "cotrans.h"          // NOLINT(build/include_subdir)
#include "ext_kalman_slam.h"  // NOLINT(build/include_subdir)
#include "landmark_map.h"     // NOLINT(build/include_subdir)

using Pose = Position2D<double>;
using TMat = TransMatrix2D<double>;
using CT   = CoTransT<double>;

struct position {
  double x;
  double y;
  double theta;
};

void printState(const EKFSLAM::EKFState& state) {
  std::cout << "State mean (mu): " << state.mu.transpose() << std::endl;
  std::cout << "State covariance (sigma): " << std::endl
            << state.sigma << std::endl;
}

void calcOdometry(Pose& pos, const EKFSLAM::Control& u) {
  double x     = pos.x;
  double y     = pos.y;
  double theta = pos.rho;

  if (std::fabs(u.w) < 1e-5) {
    // Straight motion
    pos.x   = x + u.v * u.dt * std::cos(theta);
    pos.y   = y + u.v * u.dt * std::sin(theta);
    pos.rho = AngleTool::normaliseAngleSym0(theta);
  } else {
    // Pose update for non-zero angular velocity
    pos.x   = x + u.v * std::cos(theta) * u.dt;
    pos.y   = y + u.v * std::sin(theta) * u.dt;
    pos.rho = AngleTool::normaliseAngleSym0(theta + u.w * u.dt);
  }
}

double calcDist(const Pose& pos1, const Pose& pos2 = {}) {
  double dx = pos1.x - pos2.x;
  double dy = pos1.y - pos2.y;
  return std::sqrt(dx * dx + dy * dy);
}

double getRandomDouble(double min, double max) {
  static std::random_device        rd;
  static std::mt19937              gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}
int main() {
  // Create an instance of the EKF SLAM class
  static constexpr float dt                = 0.1;    // Time step
  static constexpr float measurement_range = 500.0;  // Measurement range in mm
  Pose                   pos_robot{0, 0, 0};  // Initial position (x, y, theta)
  Pose                   pos_m1{1300, 1000, 0}, pos_m1_in_robot{};
  Pose                   pos_m2{-1300, 1000, 0}, pos_m2_in_robot{};
  Pose                   pos_m3{0, 2300, 0}, pos_m3_in_robot{};
  Pose                   pos_m4{700, 500, 0}, pos_m4_in_robot{};
  TMat                   tm_robot_in_world{};
  TMat                   tm_robot_in_world_inv{};
  EKFSLAM                ekf{}, ekf_real{};
  Eigen::Matrix3d        motionNoise;
  Eigen::Matrix2d        measurementNoise;

  // Small process noise
  motionNoise.setIdentity();
  motionNoise(0, 0) *= 2.0;   // Robot x noise in mm
  motionNoise(1, 1) *= 2.0;   // Robot y noise in mm
  motionNoise(2, 2) *= 0.10;  // Robot theta noise in radians

  // // Small measurement noise
  measurementNoise.setIdentity();
  measurementNoise(0, 0) = 5.0;   // Range noise in mm
  measurementNoise(1, 1) = 0.05;  // Bearing noise in radians

  // Optionally seed the state from a previously stored landmark map so that
  // tracking starts with already-known landmarks instead of discovering them.
  LandmarkMap prior_map;
  if (prior_map.Load("landmark_map.csv")) {
    ekf_real.SetStateFromMap(prior_map);
    std::cout << "Seeded state from landmark_map.csv (" << prior_map.size()
              << " landmarks)" << std::endl;
  }

  // Open output file and write header
  std::ofstream outFile("ekf_state_log.txt", std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file!" << std::endl;
    return 1;
  }

  // Write header — ideal (undisturbed) columns first, then noisy columns
  EKFSLAM::WriteHeader(outFile);
  int64_t timestamp = 0;  // in milliseconds

  for (float t = 0.0; t < 250.0; t += dt) {
    timestamp += static_cast<int64_t>(dt * 1000);  // Update timestamp in
                                                   // milliseconds
    double noise_v = getRandomDouble(-measurementNoise(0, 0),
                                     measurementNoise(0, 0));  // Linear
                                                               // velocity noise
    double noise_w = getRandomDouble(-measurementNoise(1, 1),
                                     measurementNoise(1, 1));  // Angular
                                                               // velocity noise
    // Simulate control input (e.g., move forward with some angular velocity)
    EKFSLAM::Control control{100.0, 0.1, dt};  // Linear velocity,
                                               // angular velocity,
                                               // time step
    EKFSLAM::Control control_real{control.v + noise_v, control.w + noise_w,
                                  control.dt};
    calcOdometry(pos_robot, control);
    CT::getTransMatrix2d(tm_robot_in_world, pos_robot);
    CT::invertTransMatrix2d(tm_robot_in_world_inv, tm_robot_in_world);
    CT::transPosition2d(tm_robot_in_world_inv, pos_m1, pos_m1_in_robot);
    CT::transPosition2d(tm_robot_in_world_inv, pos_m2, pos_m2_in_robot);
    CT::transPosition2d(tm_robot_in_world_inv, pos_m3, pos_m3_in_robot);
    CT::transPosition2d(tm_robot_in_world_inv, pos_m4, pos_m4_in_robot);

    double dist_m1 = calcDist(pos_m1_in_robot);
    double dist_m2 = calcDist(pos_m2_in_robot);
    double dist_m3 = calcDist(pos_m3_in_robot);
    double dist_m4 = calcDist(pos_m4_in_robot);

    ekf.Predict(control, motionNoise);
    ekf_real.Predict(control_real, motionNoise);

    if (dist_m1 < measurement_range) {
      EKFSLAM::Measurement meas1{
          -1,
          Eigen::Vector2d(dist_m1,
                          std::atan2(pos_m1_in_robot.y, pos_m1_in_robot.x)),
          timestamp};
      // ekf.Update(state, meas1, measurementNoise); // Update with ideal
      // measurement not needed
      ekf_real.Update(meas1, measurementNoise);
    }

    if (dist_m2 < measurement_range) {
      EKFSLAM::Measurement meas2{
          -1,
          Eigen::Vector2d(dist_m2,
                          std::atan2(pos_m2_in_robot.y, pos_m2_in_robot.x)),
          timestamp};
      // ekf.Update(state, meas2, measurementNoise); // Update with ideal
      // measurement not needed
      ekf_real.Update(meas2, measurementNoise);
    }

    if (dist_m3 < measurement_range) {
      EKFSLAM::Measurement meas3{
          -1,
          Eigen::Vector2d(dist_m3,
                          std::atan2(pos_m3_in_robot.y, pos_m3_in_robot.x)),
          timestamp};
      // ekf.Update(state, meas3, measurementNoise); // Update with ideal
      // measurement not needed
      ekf_real.Update(meas3, measurementNoise);
    }

    if (dist_m4 < measurement_range) {
      EKFSLAM::Measurement meas4{
          -1,
          Eigen::Vector2d(dist_m4,
                          std::atan2(pos_m4_in_robot.y, pos_m4_in_robot.x)),
          timestamp};
      // measurement not needed
      ekf_real.Update(meas4, measurementNoise);
    }

    std::cout << "Time: " << t << "s, Position: (" << pos_robot.x << ", "
              << pos_robot.y << "), Theta: " << pos_robot.rho * 180.0 / M_PI
              << " degrees" << "   m1: (" << pos_m1_in_robot.x << ", "
              << pos_m1_in_robot.y
              << "), Theta: " << pos_m1_in_robot.rho * 180.0 / M_PI
              << " degrees" << "   m2: (" << pos_m2_in_robot.x << ", "
              << pos_m2_in_robot.y
              << "), Theta: " << pos_m2_in_robot.rho * 180.0 / M_PI
              << " degrees" << "   m3: (" << pos_m3_in_robot.x << ", "
              << pos_m3_in_robot.y
              << "), Theta: " << pos_m3_in_robot.rho * 180.0 / M_PI
              << " degrees" << "   m4: (" << pos_m4_in_robot.x << ", "
              << pos_m4_in_robot.y
              << "), Theta: " << pos_m4_in_robot.rho * 180.0 / M_PI
              << " degrees" << "  noise: " << noise_v << std::endl;

    // Write state to file (ideal first, then noisy)
    EKFSLAM::WriteStateToFile(outFile, t, ekf, ekf_real);
  }

  // Close output file
  outFile.close();
  std::cout << "State log written to ekf_state_log.txt" << std::endl;

  // Store the final landmark estimates in a reloadable map container and
  // persist them so a later run can reload them via LandmarkMap::load().
  LandmarkMap landmark_map = ekf_real.GetLandmarkMap();
  if (landmark_map.Save("landmark_map.csv")) {
    std::cout << "Landmark map (" << landmark_map.size()
              << " landmarks) written to landmark_map.csv" << std::endl;
  } else {
    std::cerr << "Error: Could not write landmark_map.csv" << std::endl;
  }

  // Print the estimated state
  std::cout << "Estimated State: " << std::endl;

  return 0;
}
