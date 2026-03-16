#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cmath>
#include "position2d.h"
#include "ext_kalman_slam.h"
#include "cotrans.h"
#include <random>

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

void writeStateToFile(std::ofstream& file, double time, EKFSLAM& ekf,
                      const EKFSLAM::EKFState& state) {
  // Get variances and convert to standard deviations
  EKFSLAM::Variances var = ekf.getVariances(state);

  // Write time
  file << time;

  // Write robot pose (x, y, thzcc_conn_testeta)
  file << "," << state.mu(0) << "," << state.mu(1) << "," << state.mu(2);

  // Write robot standard deviations (std_x, std_y, std_theta)
  file << "," << std::sqrt(var.robot(0)) << "," << std::sqrt(var.robot(1))
       << "," << std::sqrt(state.sigma(2, 2));

  // Write landmark poses and standard deviations
  for (size_t i = 0; i < var.landmarks.size(); ++i) {
    int idx = ekf.landmarkIndex(i);
    // Landmark pose
    file << "," << state.mu(idx) << "," << state.mu(idx + 1);
    // Landmark standard deviations
    file << "," << std::sqrt(var.landmarks[i](0)) << ","
         << std::sqrt(var.landmarks[i](1));
  }

  file << std::endl;
}

void calcOdometry(position_2d& pos, const EKFSLAM::Control& u) {
  double x     = static_cast<double>(pos.x);
  double y     = static_cast<double>(pos.y);
  double theta = static_cast<double>(pos.rho);

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

float calcDist(const position_2d& pos1, const position_2d& pos2 = {}) {
  double dx = static_cast<double>(pos1.x - pos2.x);
  double dy = static_cast<double>(pos1.y - pos2.y);
  return static_cast<float>(std::sqrt(dx * dx + dy * dy));
}

double getRandomDouble(double min, double max) {
  static std::random_device        rd;
  static std::mt19937              gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}
int main() {
  // Create an instance of the EKF SLAM class
  float         dt = 0.1;                    // Time step
  position_2d   pos_robot{0, 0, 0.0f};       // Initial position (x, y, theta)
  position_2d   pos_robot_real{0, 0, 0.0f};  // Initial position (x, y, theta)
  position_2d   pos_m1{1300, 1000, 0.0f}, pos_m1_in_robot{};
  position_2d   pos_m2{-1300, 1000, 0.0f}, pos_m2_in_robot{};
  position_2d   pos_m3{700, 500, 0.0f}, pos_m3_in_robot{};
  position_2d   pos_m4{0, 2300, 0.0f}, pos_m4_in_robot{};
  transMatrix2d tm_robot_in_world;
  transMatrix2d tm_robot_in_world_inv;
  EKFSLAM       ekf, ekf_real;
  EKFSLAM::EKFState state{}, state_real{};
  Eigen::Matrix3d   motionNoise;
  Eigen::Matrix2d   measurementNoise;

  // Initialize state with robot at origin and two landmarks
  state.mu      = Eigen::VectorXd::Zero(EKFSLAM::STATE_SIZE);  // [x, y, theta,
                                                               // lm1_x, lm1_y,
                                                               // lm2_x, lm2_y,
                                                               // lm3_x, lm3_y,
                                                               // lm4_x, lm4_y]
  state_real.mu = state.mu;
  state.sigma =
      Eigen::MatrixXd::Identity(EKFSLAM::STATE_SIZE, EKFSLAM::STATE_SIZE) * 0.1;
  state_real.sigma = state.sigma;
  // Small process noise
  motionNoise.setIdentity();
  motionNoise(0, 0) *= 1.0;                 // Robot x noise
  motionNoise(1, 1) *= 1.0;                 // Robot y noise
  motionNoise(2, 2) *= 0.5 * M_PI / 180.0;  // Robot theta noise

  // // Small measurement noise
  // measurementNoise.setIdentity();
  // measurementNoise *= 10.0;

  // Open output file and write header
  std::ofstream outFile("ekf_state_log.txt", std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file!" << std::endl;
    return 1;
  }

  // Write header
  outFile << "time,robot_x,robot_y,robot_theta,robot_std_x,robot_std_y,"
             "robot_std_theta";
  for (size_t i = 0; i < EKFSLAM::NUM_LANDMARKS; ++i) {  // 2 landmarks
    outFile << ",lm" << i << "_x,lm" << i << "_y,lm" << i << "_std_x,lm" << i
            << "_std_y";
  }
  outFile << std::endl;
  bool saw_m1 = false;
  bool saw_m2 = false;
  bool saw_m3 = false;
  bool saw_m4 = false;

  for (float t = 0.0; t < 150.0; t += 0.1) {
    double noise_v = getRandomDouble(-3.0, 3.0);    // Linear velocity noise
    double noise_w = getRandomDouble(-0.05, 0.05);  // Angular velocity noise
    // Simulate control input (e.g., move forward with some angular velocity)
    EKFSLAM::Control control{100.0, 0.1, dt};  // Linear velocity,
                                               // angular velocity,
                                               // time step
    EKFSLAM::Control control_real{control.v + noise_v, control.w + noise_w,
                                  control.dt};
    calcOdometry(pos_robot, control);
    CoTrans::getTransMatrix2d(tm_robot_in_world, &pos_robot);
    CoTrans::invertTransMatrix2d(tm_robot_in_world, tm_robot_in_world_inv);
    CoTrans::transPosition2d(tm_robot_in_world_inv, &pos_m1, &pos_m1_in_robot);
    CoTrans::transPosition2d(tm_robot_in_world_inv, &pos_m2, &pos_m2_in_robot);
    CoTrans::transPosition2d(tm_robot_in_world_inv, &pos_m3, &pos_m3_in_robot);
    CoTrans::transPosition2d(tm_robot_in_world_inv, &pos_m4, &pos_m4_in_robot);

    float dist_m1 = calcDist(pos_m1_in_robot);
    float dist_m2 = calcDist(pos_m2_in_robot);
    float dist_m3 = calcDist(pos_m3_in_robot);
    float dist_m4 = calcDist(pos_m4_in_robot);

    ekf.predict(state, control, motionNoise);
    ekf_real.predict(state_real, control_real, motionNoise);

    if (dist_m1 < 500.0f) {
      if (!saw_m1) {
        EKFSLAM::Measurement meas1{
            0, Eigen::Vector2d(
                   dist_m1, std::atan2(pos_m1_in_robot.y, pos_m1_in_robot.x))};
        ekf.update(state, meas1, measurementNoise);
        ekf_real.update(state_real, meas1, measurementNoise);
        saw_m1 = true;
      }
    } else {
      saw_m1 = false;
    }

    if (dist_m2 < 500.0f) {
      if (!saw_m2) {
        EKFSLAM::Measurement meas2{
            1, Eigen::Vector2d(
                   dist_m2, std::atan2(pos_m2_in_robot.y, pos_m2_in_robot.x))};
        ekf.update(state, meas2, measurementNoise);
        ekf_real.update(state_real, meas2, measurementNoise);
        saw_m2 = true;
      }
    } else {
      saw_m2 = false;
    }

    if (dist_m3 < 500.0f) {
      if (!saw_m3) {
        EKFSLAM::Measurement meas3{
            2, Eigen::Vector2d(
                   dist_m3, std::atan2(pos_m3_in_robot.y, pos_m3_in_robot.x))};
        ekf.update(state, meas3, measurementNoise);
        ekf_real.update(state_real, meas3, measurementNoise);
        saw_m3 = true;
      }
    } else {
      saw_m3 = false;
    }

    if (dist_m4 < 500.0f) {
      if (!saw_m4) {
        EKFSLAM::Measurement meas4{
            3, Eigen::Vector2d(
                   dist_m4, std::atan2(pos_m4_in_robot.y, pos_m4_in_robot.x))};
        ekf.update(state, meas4, measurementNoise);
        ekf_real.update(state_real, meas4, measurementNoise);
        saw_m4 = true;
      }
    } else {
      saw_m4 = false;
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

    // Write state to file
    // writeStateToFile(outFile, t, ekf, state);
    writeStateToFile(outFile, t, ekf_real, state_real);
  }

  // Close output file
  outFile.close();
  std::cout << "State log written to ekf_state_log.txt" << std::endl;

  // Print the estimated state
  std::cout << "Estimated State: " << std::endl;

  return 0;
}