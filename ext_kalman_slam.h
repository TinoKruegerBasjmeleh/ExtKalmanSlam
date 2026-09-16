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
#ifndef EXT_KALMAN_SLAM_H_
#define EXT_KALMAN_SLAM_H_

#include <Eigen/Dense>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "angle_tool.h"    // NOLINT(build/include_subdir)
#include "cotrans.h"       // NOLINT(build/include_subdir)
#include "landmark_map.h"  // NOLINT(build/include_subdir)

class EKFSLAM {
 public:
  static constexpr size_t  NUM_LANDMARKS      = 4;
  static constexpr size_t  STATE_SIZE         = 3 + 2 * NUM_LANDMARKS;
  static constexpr double  MAX_DIST_THRESHOLD = 500;  // Maximum distance in mm
  // Minimum time difference in ms for detecting loop closure
  static constexpr int64_t MIN_TIMEDIFF_2_DETECT_LOOP_CLOSURE = 1000;

  struct EKFState {
    Eigen::VectorXd mu{};     // State mean
    Eigen::MatrixXd sigma{};  // Covariance
  };
  struct Control {
    double v{};   // Linear velocity
    double w{};   // Angular velocity
    double dt{};  // Time step
  };
  struct Measurement {
    int             id{-1};       // Landmark ID
    Eigen::Vector2d z{};          // Measurement (range, bearing)
    int64_t         timestamp{};  // Timestamp of the measurement
  };

  struct Variances {
    Eigen::Vector3d              robot;      // Robot x, y, theta variance
    std::vector<Eigen::Vector2d> landmarks;  // Landmark x, y variances
  };

  void Predict(const Control& u, const Eigen::Matrix3d& motionNoise) {
    Predict(state_, u, motionNoise);
  }

  bool Update(Measurement& meas, const Eigen::Matrix2d& Q) {
    int assoziated_id = -1;
    // make data association
    assoziated_id     = GetAssociatedLandmarkId(meas);
    if (assoziated_id == -1) return false;
    if (assoziated_id < 0 || static_cast<size_t>(assoziated_id) >= map_.size())
      return false;

    Landmark lm = map_[static_cast<size_t>(assoziated_id)];

    meas.id     = assoziated_id;
    // update the kalman state based on observation
    bool detect_loop_closure =
        abs(meas.timestamp - lm.timestamp) > MIN_TIMEDIFF_2_DETECT_LOOP_CLOSURE
            ? true
            : false;
    Update(state_, meas, Q, detect_loop_closure);

    // update the landmark inside map from the state
    Eigen::Vector2d landmark_pos = LandmarkPose(state_, assoziated_id);
    Variances       var          = GetVariances(state_);
    // write the refreshed estimate back into the map (single source of truth)

    lm.timestamp                 = meas.timestamp;
    lm.x                         = landmark_pos[0];
    lm.y                         = landmark_pos[1];
    lm.std_x                     = sqrt(var.landmarks[assoziated_id](0));
    lm.std_y                     = sqrt(var.landmarks[assoziated_id](1));
    map_[assoziated_id]          = lm;
    return true;
  }

  Position2D<double> GetRobotPose() {
    Eigen::Vector3d pose = state_.mu.segment<3>(0);
    return Position2D<double>{pose[0], pose[1], pose[2]};
  }

  Eigen::Vector2d LandmarkPose(int landmark_id) {
    return LandmarkPose(state_, landmark_id);
  }
  Variances                   GetVariances() { return GetVariances(state_); }

  /**
   * @brief RobotLandmarkCrossCovariance
   * The 3x2 covariance block coupling the robot pose to one landmark. Exposed
   * so that callers (and tests) can verify the filter actually maintains these
   * correlations, which is what distinguishes EKF-SLAM from a set of
   * independent per-landmark filters.
   */
  Eigen::Matrix<double, 3, 2> RobotLandmarkCrossCovariance(int landmark_id) {
    return state_.sigma.block<3, 2>(0, LandmarkIndex(landmark_id));
  }

  /**
   * @brief GetAssociatedLandmarkId
   * Perform data association for a given measurement.
   * @param meas The measurement to associate. The measurement is relative
   * to the robot's current pose and in range-bearing (radian, mm)
   * representation.
   * @return The ID of the associated landmark, or -1 if no suitable match is
   * found.
   */
  int GetAssociatedLandmarkId(Measurement& meas) {
    const double          r   = meas.z[0];
    const double          phi = meas.z[1];
    // Robot-relative landmark position, using the same (cos, sin) convention
    // as InitializeLandmark() and the measurement model.
    Position2D<double>    lm_pos_robot_relative{r * std::cos(phi),
                                                r * std::sin(phi), 0.0};

    Position2D<double>    lm_pos_global{}, robot_global{};
    TransMatrix2D<double> tm_robot_in_world{};

    robot_global = GetRobotPose();
    // 1. Transform the measurement into global coordinates.
    CoTransT<double>::getTransMatrix2d(tm_robot_in_world, robot_global);
    CoTransT<double>::transPosition2d(tm_robot_in_world, lm_pos_robot_relative,
                                      lm_pos_global);

    // 2. Match against the current state estimates (single source of truth),
    //    keeping the nearest already-observed landmark within the gate.
    int    best_id    = -1;
    int    first_free = -1;
    double best_dist  = MAX_DIST_THRESHOLD;
    for (int id = 0; id < static_cast<int>(NUM_LANDMARKS); ++id) {
      if (!CheckIfLandmarkObserved(state_, id)) {
        if (first_free == -1) first_free = id;
        continue;
      }
      Eigen::Vector2d p = LandmarkPose(state_, id);
      double          dist =
          std::hypot(lm_pos_global.x - p.x(), lm_pos_global.y - p.y());
      if (dist < best_dist) {
        best_dist = dist;
        best_id   = id;
      }
    }

    // 3. Re-observe the matched landmark, otherwise allocate a free slot.
    return best_id != -1 ? best_id : first_free;
  }

  LandmarkMap GetLandmarkMap() { return map_; }

  void        SetStateFromMap(const LandmarkMap& map) {
    for (const Landmark& lm : map.GetAll()) {
      if (lm.id >= 0 && static_cast<size_t>(lm.id) < NUM_LANDMARKS) {
        map_[lm.id] = lm;
      }
    }
    SetStateFromMap(state_, map);
  }
  void SetInitialPos(const Eigen::Vector3d& initialPos,
                     const Eigen::Matrix3d& initialCov) {
    SetInitialPos(state_, initialPos, initialCov);
  }

  inline int LandmarkIndex(int id) { return 3 + 2 * id; }

  /****************************************************************************
   * @brief WriteStateBlock
   * Write this instance's pose + landmark estimates as CSV fields (each field
   * prefixed with a comma). Variances are written as covariance diagonal
   * elements; the consumer (animate_ekf.py) applies sqrt() before use.
   * @param out the output stream to write to
   * ***************************************************************************/
  void       WriteStateBlock(std::ofstream& out) {
    Position2D<double> pose = GetRobotPose();
    Variances          var  = GetVariances();
    out << "," << pose.x << "," << pose.y << "," << pose.rho << ","
        << sqrt(var.robot(0)) << "," << sqrt(var.robot(1));
    for (size_t i = 0; i < NUM_LANDMARKS; ++i) {
      Eigen::Vector2d p = LandmarkPose(static_cast<int>(i));
      out << "," << p.x() << "," << p.y() << "," << sqrt(var.landmarks[i](0))
          << "," << sqrt(var.landmarks[i](1));
    }
  }

  /****************************************************************************
   * @brief WriteStateToFile
   * Write a full CSV row: time, ideal state block, then noisy state block.
   * @param out the output stream to write to
   * @param time the current simulation time
   * @param ideal the EKF driven by noise-free control
   * @param noisy the EKF driven by noisy control and corrected by measurements
   * ***************************************************************************/
  static void WriteStateToFile(std::ofstream& out, double time, EKFSLAM& ideal,
                               EKFSLAM& noisy) {
    out << time;
    ideal.WriteStateBlock(out);
    noisy.WriteStateBlock(out);
    out << "\n";
  }

  /****************************************************************************
   * @brief WriteHeader
   * Write the CSV header — ideal (undisturbed) columns first, then noisy
   * columns (prefixed with "noisy_"). Column layout must stay in sync with
   * animate_ekf.py.
   * @param out the output stream to write to
   * ***************************************************************************/
  static void WriteHeader(std::ofstream& out) {
    auto writeLmHeader = [&](const std::string& prefix) {
      out << "," << prefix << "robot_x" << "," << prefix << "robot_y" << ","
          << prefix << "robot_theta" << "," << prefix << "robot_std_x" << ","
          << prefix << "robot_std_y";
      for (size_t i = 0; i < NUM_LANDMARKS; ++i) {
        out << "," << prefix << "lm" << i << "_x" << "," << prefix << "lm" << i
            << "_y" << "," << prefix << "lm" << i << "_std_x" << "," << prefix
            << "lm" << i << "_std_y";
      }
    };
    out << "time";
    writeLmHeader("");
    writeLmHeader("noisy_");
    out << "\n";
  }

 private:
  LandmarkMap map_{};
  EKFState    state_{};

  void        Predict(EKFState& state, const Control& u,
                      const Eigen::Matrix3d& motionNoise) {
    static_assert(STATE_SIZE == 3 + 2 * NUM_LANDMARKS,
                  "Example assumes 4 landmarks");

    // 1. Predict landmark pose  and  robot pose
    Eigen::Vector3d pose      = state.mu.segment<3>(0);
    Eigen::Vector3d pose_pred = MotionModel(pose, u);
    state.mu.segment<3>(0)    = pose_pred;

    // 2. Build full Jacobian G_t
    Eigen::MatrixXd G   = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    G.block<3, 3>(0, 0) = MotionJacobian(pose, u);

    // 3. Motion noise in full state space
    Eigen::MatrixXd R   = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
    R.block<3, 3>(0, 0) = motionNoise;

    // 4. Covariance prediction
    // Todo: Update only the covariance of the robot and the cross-covariances
    // with landmarks, not the entire matrix
    state.sigma         = G * state.sigma * G.transpose() + R;
  }
  void Update(EKFState& state, Measurement& meas, const Eigen::Matrix2d& Q,
              bool detect_loop_closure = false) {
    int             landmarkId = meas.id;
    Eigen::Vector2d z          = meas.z;

    if (!CheckIfLandmarkObserved(state, landmarkId)) {
      InitializeLandmark(state, landmarkId, z, Q);
      return;
    }
    // 1. Predict measurement
    Eigen::Vector2d z_hat = PredictLandmarkMeasurement(state.mu, landmarkId);
    // 2. Measurement residual
    Eigen::Vector2d y     = z - z_hat;
    y(1)                  = AngleTool::normaliseAngleSym0(y(1));

    // 3. Measurement Jacobian
    Eigen::Matrix<double, 2, 5> H(2, 5);
    H.setZero();
    H = MeasurementJacobian(state.mu, landmarkId);
    const int                   lm_idx = LandmarkIndex(landmarkId);
    Eigen::Matrix<double, 5, 5> sigma_sub;
    sigma_sub.setZero();
    // Extract the relevant submatrix of the covariance for the robot (3x3
    // block)
    sigma_sub.block<3, 3>(0, 0) = state.sigma.block<3, 3>(0, 0);
    // Extract the relevant submatrix of the covariance for the landmark (2x2
    // block)
    sigma_sub.block<2, 2>(3, 3) = state.sigma.block<2, 2>(lm_idx, lm_idx);

    // Extract the cross-covariance between the robot and the landmark (3x2
    // block)
    // sigma_sub.block<3, 2>(0, 3) = state.sigma.block<3, 2>(0, lm_idx);
    // sigma_sub.block<2, 3>(3, 0) = state.sigma.block<2, 3>(lm_idx, 0);
    // Now sigma_sub contains the relevant 5x5 submatrix for the robot and the
    // landmark
    // 4. Innovation covariance
    Eigen::Matrix2d S = H * sigma_sub * H.transpose() + Q;  // Is Q
                                                            // ododmetey
                                                            // and measurement
                                                            // noise? Yes, it's
                                                            // the measurement
                                                            // noise covariance
    // 5. Kalman gain
    Eigen::Matrix<double, 5, 2> K = sigma_sub * H.transpose() * S.inverse();

    // 6. State update
    state.mu(0) += K(0, 0) * y(0) + K(0, 1) * y(1);
    state.mu(1) += K(1, 0) * y(0) + K(1, 1) * y(1);
    state.mu(2) += K(2, 0) * y(0) + K(2, 1) * y(1);
    state.mu(LandmarkIndex(landmarkId)) += K(3, 0) * y(0) + K(3, 1) * y(1);
    state.mu(LandmarkIndex(landmarkId) + 1) += K(4, 0) * y(0) + K(4, 1) * y(1);

    state.mu(2) = AngleTool::normaliseAngleSym0(state.mu(2));

    if (!detect_loop_closure) {
      return;
    }

    // 7. Covariance update (Joseph form recommended)
    Eigen::MatrixXd I             = Eigen::MatrixXd::Identity(5, 5);
    sigma_sub                     = (I - K * H) * sigma_sub;
    state.sigma.block<3, 3>(0, 0) = sigma_sub.block<3, 3>(0, 0);
    state.sigma.block<2, 2>(LandmarkIndex(landmarkId),
                            LandmarkIndex(landmarkId)) =
        sigma_sub.block<2, 2>(3, 3);
  }

  void SetInitialPos(EKFState& state, const Eigen::Vector3d& initialPos,
                     const Eigen::Matrix3d& initialCov) {
    state.mu.head<3>()            = initialPos;
    state.sigma.block<3, 3>(0, 0) = initialCov;
  }

  /****************************************************************************
   * @brief GetVariances
   * Extract the x and y variances from the covariance matrix for the robot
   * pose and all landmarks.
   * @param state the current EKF state
   * @return Variances struct containing robot and landmark x,y variances
   * ***************************************************************************/
  Variances GetVariances(const EKFState& state) {
    Variances var;

    // Extract robot pose variance (x, y)
    var.robot(0) = state.sigma(0, 0);  // x variance
    var.robot(1) = state.sigma(1, 1);  // y variance
    var.robot(2) = state.sigma(2, 2);  // theta variance

    // Extract landmark variances
    var.landmarks.resize(NUM_LANDMARKS);
    for (size_t i = 0; i < NUM_LANDMARKS; ++i) {
      int idx             = LandmarkIndex(i);
      var.landmarks[i](0) = state.sigma(idx, idx);  // landmark x variance
      var.landmarks[i](1) = state.sigma(idx + 1, idx + 1);  // landmark y
                                                            // variance
    }

    return var;
  }
  /****************************************************************************
   * @brief GetMapFromState
   * Build a LandmarkMap holding the current landmark estimates (position,
   * standard deviations and observed flag) so the already-existing landmarks
   * can be stored / serialised.
   * @param state the current EKF state
   * @return a LandmarkMap populated from the state
   * ***************************************************************************/
  LandmarkMap GetMapFromState(const EKFState& state) {
    LandmarkMap map;
    Variances   var = GetVariances(state);
    for (size_t i = 0; i < NUM_LANDMARKS; ++i) {
      if (!CheckIfLandmarkObserved(state, i)) {
        map.AddLandmark(static_cast<int>(i), std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max());
        continue;
      }
      int             id  = static_cast<int>(i);
      Eigen::Vector2d pos = LandmarkPose(state, id);
      map.AddLandmark(id, pos.x(), pos.y(), std::sqrt(var.landmarks[i](0)),
                      std::sqrt(var.landmarks[i](1)),
                      CheckIfLandmarkObserved(state, id));
    }
    return map;
  }
  /****************************************************************************
   * @brief SetStateFromMap
   * Seed the EKF state at the beginning of tracking from a previously stored
   * LandmarkMap. Each landmark's position is written into the state mean and
   * its 2x2 covariance block is set from the stored standard deviations. Only
   * ids that fit within the fixed state layout are applied. No run-time update
   * of the map is performed after this.
   * @param state the EKF state to seed (mean and covariance)
   * @param map the landmark map to read from
   * ***************************************************************************/
  void SetStateFromMap(EKFState& state, const LandmarkMap& map) {
    for (const Landmark& lm : map.GetAll()) {
      if (lm.id < 0 || static_cast<size_t>(lm.id) >= NUM_LANDMARKS) {
        continue;
      }
      int idx           = LandmarkIndex(lm.id);
      state.mu(idx)     = lm.x;
      state.mu(idx + 1) = lm.y;

      // Reset the landmark covariance block to a diagonal built from the
      // stored standard deviations (no cross-covariances), consistent with
      // InitializeLandmark().
      state.sigma.block<2, 2>(idx, idx).setZero();
      state.sigma(idx, idx)         = lm.std_x * lm.std_x;
      state.sigma(idx + 1, idx + 1) = lm.std_y * lm.std_y;
    }
  }
  // begin private function definition
  Eigen::Vector3d MotionModel(const Eigen::Vector3d& pose, const Control& u) {
    double          x     = pose(0);
    double          y     = pose(1);
    double          theta = pose(2);
    Eigen::Vector3d new_pose{};

    if (std::fabs(u.w) < 1e-5) {
      // Straight motion
      new_pose(0) = x + u.v * u.dt * std::cos(theta);
      new_pose(1) = y + u.v * u.dt * std::sin(theta);
      new_pose(2) = AngleTool::normaliseAngleSym0(theta);
    } else {
      double v_w    = u.v / u.w;
      double dtheta = u.w * u.dt;
      // Pose update for non-zero angular velocity
      new_pose(0) = x - v_w * std::sin(theta) + v_w * std::sin(theta + dtheta);
      new_pose(1) = y + v_w * std::cos(theta) - v_w * std::cos(theta + dtheta);
      new_pose(2) = AngleTool::normaliseAngleSym0(theta + dtheta);
    }

    return new_pose;
  }

  Eigen::Matrix3d MotionJacobian(const Eigen::Vector3d& pose,
                                 const Control&         u) {
    Eigen::Matrix3d G     = Eigen::Matrix3d::Identity();

    double          theta = pose(2);

    if (std::fabs(u.w) < 1e-5) {
      // Straight-line motion
      G(0, 2) = -u.v * u.dt * std::sin(theta);
      G(1, 2) = u.v * u.dt * std::cos(theta);
      return G;
    }

    double v_w    = u.v / u.w;
    double dtheta = u.w * u.dt;

    G(0, 2)       = -v_w * std::cos(theta) + v_w * std::cos(theta + dtheta);
    G(1, 2)       = -v_w * std::sin(theta) + v_w * std::sin(theta + dtheta);

    return G;
  }

  /****************************************************************************
   * @brief RangeBearingObservation
   * Calculate the expected observation (range and bearing) for a given landmark
   * based on the current state estimate in global coordinates.
   * @param mu the current state mean vector
   * @param landmarkId the ID of the landmark to observe
   * @param zj the actual measurement (range, bearing) for the landmark
   * @return the expected observation (range, bearing) for the landmark
   * ***************************************************************************/

  Eigen::Vector2d RangeBearingObservation(const Eigen::VectorXd& mu,
                                          const Eigen::Vector2d& zj) {
    // pose of the robot
    double          x     = mu(0);
    double          y     = mu(1);
    double          theta = mu(2);

    Eigen::Vector2d z_observed{};

    z_observed(0) = x + zj(0) * std::cos(zj(1) + theta);
    z_observed(1) = y + zj(0) * std::sin(zj(1) + theta);

    return z_observed;
  }

  /****************************************************************************
   * @brief PredictLandmarkMeasurement
   * Calculate the expected measurement (range and bearing) for a given landmark
   * based on the current state estimate.
   * @param mu the current state mean vector
   * @param landmarkId the ID of the landmark to observe
   * @return the expected measurement (range, bearing) for the individual
   * landmark
   * ***************************************************************************/
  Eigen::Vector2d PredictLandmarkMeasurement(const Eigen::VectorXd& mu,
                                             int landmarkId) {
    int             idx   = LandmarkIndex(landmarkId);

    // pose of the robot
    double          x     = mu(0);
    double          y     = mu(1);
    double          theta = mu(2);

    // point of the landmark
    double          mx    = mu(idx);
    double          my    = mu(idx + 1);

    Eigen::Vector2d z_hat;

    double          dx = mx - x;
    double          dy = my - y;

    z_hat(0)           = std::sqrt(dx * dx + dy * dy);                // range
    z_hat(1) = AngleTool::normaliseAngleSym0(atan2(dy, dx) - theta);  // bearing

    return z_hat;
  }

  Eigen::Matrix<double, 2, 5> MeasurementJacobian(const Eigen::VectorXd& mu,
                                                  int landmarkId) {
    Eigen::Matrix<double, 2, 5> H;
    H.setZero();

    int    idx    = LandmarkIndex(landmarkId);

    double x      = mu(0);
    double y      = mu(1);

    double mx     = mu(idx);
    double my     = mu(idx + 1);

    double dx     = mx - x;
    double dy     = my - y;

    double q      = dx * dx + dy * dy;
    double sqrt_q = std::sqrt(q);

    // w.r.t robot pose
    H(0, 0)       = -dx / sqrt_q;
    H(0, 1)       = -dy / sqrt_q;
    H(0, 2)       = 0.0;

    H(1, 0)       = dy / q;
    H(1, 1)       = -dx / q;
    H(1, 2)       = -1.0;

    // w.r.t landmark
    H(0, 3)       = dx / sqrt_q;
    H(0, 4)       = dy / sqrt_q;

    H(1, 3)       = -dy / q;
    H(1, 4)       = dx / q;

    return H;
  }
  Eigen::Vector2d LandmarkPose(const EKFState& state, int landmarkId) {
    int    idx = LandmarkIndex(landmarkId);
    double mx  = state.mu(idx);
    double my  = state.mu(idx + 1);
    return Eigen::Vector2d(mx, my);
  }

  bool CheckIfLandmarkObserved(const EKFState& state, int landmarkId) {
    int idx = LandmarkIndex(landmarkId);
    return (fabs(state.mu(idx)) > 1.0 || fabs(state.mu(idx + 1)) > 1.0);
  }

  void InitializeLandmark(EKFState& state, int landmarkId,
                          const Eigen::Vector2d& z, const Eigen::Matrix2d& Q) {
    // z = [range, bearing] measurement
    double             r     = z(0);  // range
    double             phi   = z(1);  // bearing

    double             x     = state.mu(0);
    double             y     = state.mu(1);
    double             theta = state.mu(2);

    int                idx   = LandmarkIndex(landmarkId);

    // Initialize landmark position in global coordinates using the
    // double-precision templated CoTransT (the legacy float CoTrans
    // truncates offsets to int inside getOffset).
    Position2D<double> robot_pose{x, y, theta};
    Position2D<double> landmark_pose{r * std::cos(phi), r * std::sin(phi), 0.0};
    Position2D<double> landmark_pose_in_world{};

    TransMatrix2D<double> tm_robot_in_world{};
    CoTransT<double>::getTransMatrix2d(tm_robot_in_world, robot_pose);
    CoTransT<double>::transPosition2d(tm_robot_in_world, landmark_pose,
                                      landmark_pose_in_world);

    double lm_x       = landmark_pose_in_world.x;
    double lm_y       = landmark_pose_in_world.y;

    state.mu(idx)     = lm_x;
    state.mu(idx + 1) = lm_y;

    // Initialize landmark covariance
    state.sigma.block<2, 2>(idx, idx) =
        Q + state.sigma.block<2, 2>(0, 0);  // Add robot pose uncertainty to
                                            // landmark covariance
  }

 public:
  explicit EKFSLAM() {
    state_.mu = Eigen::VectorXd::Zero(EKFSLAM::STATE_SIZE);  // [x, y, theta,
                                                             // lm1_x, lm1_y,
                                                             // lm2_x, lm2_y,
                                                             // lm3_x, lm3_y,
                                                             // lm4_x, lm4_y]
    state_.sigma =
        Eigen::MatrixXd::Identity(EKFSLAM::STATE_SIZE, EKFSLAM::STATE_SIZE) *
        1e-3;
    map_ = GetMapFromState(state_);
  }
};

#endif  // EXT_KALMAN_SLAM_H_
