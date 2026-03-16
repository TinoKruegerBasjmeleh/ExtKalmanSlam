#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

#include "angle_tool.h"
#include "cotrans.h"

class EKFSLAM {
 public:
  static constexpr size_t NUM_LANDMARKS = 4;
  static constexpr size_t STATE_SIZE    = 3 + 2 * NUM_LANDMARKS;

  struct EKFState {
    Eigen::VectorXd mu;     // State mean
    Eigen::MatrixXd sigma;  // Covariance
  };
  struct Control {
    double v;   // Linear velocity
    double w;   // Angular velocity
    double dt;  // Time step
  };
  struct Measurement {
    int             id;  // Landmark ID
    Eigen::Vector2d z;   // Measurement (range, bearing)
  };

  struct Variances {
    Eigen::Vector3d              robot;      // Robot x, y, theta variance
    std::vector<Eigen::Vector2d> landmarks;  // Landmark x, y variances
  };

  Eigen::Vector3d motionModel(const Eigen::Vector3d& pose, const Control& u) {
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

  Eigen::Matrix3d motionJacobian(const Eigen::Vector3d& pose,
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

  void predict(EKFState& state, const Control& u,
               const Eigen::Matrix3d& motionNoise) {
    static_assert(STATE_SIZE == 3 + 2 * NUM_LANDMARKS,
                  "Example assumes 4 landmarks");

    // 1. Predict landmark pose  and  robot pose
    Eigen::Vector3d pose      = state.mu.segment<3>(0);
    Eigen::Vector3d pose_pred = motionModel(pose, u);
    state.mu.segment<3>(0)    = pose_pred;

    // 2. Build full Jacobian G_t
    Eigen::MatrixXd G   = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
    G.block<3, 3>(0, 0) = motionJacobian(pose, u);

    // 3. Motion noise in full state space
    Eigen::MatrixXd R   = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
    R.block<3, 3>(0, 0) = motionNoise;

    // 4. Covariance prediction
    // Todo: Update only the covariance of the robot and the cross-covariances
    // with landmarks, not the entire matrix
    state.sigma         = G * state.sigma * G.transpose() + R;
  }

  inline int      landmarkIndex(int id) { return 3 + 2 * id; }

  /****************************************************************************
   * @brief rangeBearingObservation
   * Calculate the expected observation (range and bearing) for a given landmark
   * based on the current state estimate in global coordinates.
   * @param mu the current state mean vector
   * @param landmarkId the ID of the landmark to observe
   * @param zj the actual measurement (range, bearing) for the landmark
   * @return the expected observation (range, bearing) for the landmark
   * ***************************************************************************/

  Eigen::Vector2d rangeBearingObservation(const Eigen::VectorXd& mu,
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
   * @brief predictLandmarkMeasurement
   * Calculate the expected measurement (range and bearing) for a given landmark
   * based on the current state estimate.
   * @param mu the current state mean vector
   * @param landmarkId the ID of the landmark to observe
   * @return the expected measurement (range, bearing) for the individual
   * landmark
   * ***************************************************************************/
  Eigen::Vector2d predictLandmarkMeasurement(const Eigen::VectorXd& mu,
                                             int landmarkId) {
    int             idx   = landmarkIndex(landmarkId);

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

  Eigen::Matrix<double, 2, 5> measurementJacobian(const Eigen::VectorXd& mu,
                                                  int landmarkId) {
    Eigen::Matrix<double, 2, 5> H;
    H.setZero();

    int    idx    = landmarkIndex(landmarkId);

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
  Eigen::Vector2d landmarkPose(const EKFState& state, int landmarkId) {
    int    idx = landmarkIndex(landmarkId);
    double mx  = state.mu(idx);
    double my  = state.mu(idx + 1);
    return Eigen::Vector2d(mx, my);
  }

  bool checkIfLandmarkObserved(const EKFState& state, int landmarkId) {
    int idx = landmarkIndex(landmarkId);
    return (fabs(state.mu(idx)) > 1.0 || fabs(state.mu(idx + 1)) > 1.0);
  }

  void initializeLandmark(EKFState& state, int landmarkId,
                          const Eigen::Vector2d& z, const Eigen::Matrix2d& Q) {
    // z = [range, bearing] measurement
    double      r     = z(0);  // range
    double      phi   = z(1);  // bearing

    double      x     = state.mu(0);
    double      y     = state.mu(1);
    double      theta = state.mu(2);

    int         idx   = landmarkIndex(landmarkId);

    // Initialize landmark position in global coordinates
    position_2d robot_pose{static_cast<float>(x), static_cast<float>(y),
                           static_cast<float>(theta)};
    position_2d landmark_pose =
        position_2d{static_cast<float>(r * std::cos(phi)),
                    static_cast<float>(r * std::sin(phi)), 0.0};
    position_2d   landmark_pose_in_world{};

    transMatrix2d tm_robot_in_world;
    CoTrans::getTransMatrix2d(tm_robot_in_world, &robot_pose);

    CoTrans::transPosition2d(tm_robot_in_world, &landmark_pose,
                             &landmark_pose_in_world);

    double lm_x       = landmark_pose_in_world.x;
    double lm_y       = landmark_pose_in_world.y;

    state.mu(idx)     = lm_x;
    state.mu(idx + 1) = lm_y;
    // Initialize landmark covariance
    state.sigma.block<2, 2>(idx, idx) =
        Q + state.sigma.block<2, 2>(0, 0);  // Add robot pose uncertainty to
                                            // landmark covariance
  }

  void update(EKFState& state, Measurement& meas, const Eigen::Matrix2d& Q) {
    const int       n          = state.mu.size();

    int             landmarkId = meas.id;
    Eigen::Vector2d z          = meas.z;

    if (!checkIfLandmarkObserved(state, landmarkId)) {
      initializeLandmark(state, landmarkId, z, Q);
    }
    // 1. Predict measurement
    Eigen::Vector2d z_hat = predictLandmarkMeasurement(state.mu, landmarkId);
    // 2. Measurement residual
    Eigen::Vector2d y     = z - z_hat;
    y(1)                  = AngleTool::normaliseAngleSym0(y(1));

    // 3. Measurement Jacobian
    Eigen::Matrix<double, 2, 5> H(2, 5);
    H.setZero();
    H = measurementJacobian(state.mu, landmarkId);

    Eigen::Matrix<double, 5, 5> sigma_sub      = state.sigma.block<5, 5>(0, 0);
    Eigen::Matrix2d             sigma_landmark = state.sigma.block<2, 2>(
        landmarkIndex(landmarkId), landmarkIndex(landmarkId));
    sigma_sub.block<2, 2>(3, 3)   = sigma_landmark;
    // 4. Innovation covariance
    Eigen::Matrix2d             S = H * sigma_sub * H.transpose() + Q;

    // 5. Kalman gain
    Eigen::Matrix<double, 5, 2> K = sigma_sub * H.transpose() * S.inverse();

    // 6. State update
    state.mu(0) += K(0, 0) * y(0) + K(0, 1) * y(1);
    state.mu(1) += K(1, 0) * y(0) + K(1, 1) * y(1);
    state.mu(2) += K(2, 0) * y(0) + K(2, 1) * y(1);
    state.mu(landmarkIndex(landmarkId)) += K(3, 0) * y(0) + K(3, 1) * y(1);
    state.mu(landmarkIndex(landmarkId) + 1) += K(4, 0) * y(0) + K(4, 1) * y(1);

    // 7. Covariance update (Joseph form recommended)
    Eigen::MatrixXd I             = Eigen::MatrixXd::Identity(5, 5);
    sigma_sub                     = (I - K * H) * sigma_sub;
    state.sigma.block<3, 3>(0, 0) = sigma_sub.block<3, 3>(0, 0);
    state.sigma.block<2, 2>(landmarkIndex(landmarkId),
                            landmarkIndex(landmarkId)) =
        sigma_sub.block<2, 2>(3, 3);
  }

  /****************************************************************************
   * @brief getVariances
   * Extract the x and y variances from the covariance matrix for the robot
   * pose and all landmarks.
   * @param state the current EKF state
   * @return Variances struct containing robot and landmark x,y variances
   * ***************************************************************************/
  Variances getVariances(const EKFState& state) {
    Variances var;

    // Extract robot pose variance (x, y)
    var.robot(0) = state.sigma(0, 0);  // x variance
    var.robot(1) = state.sigma(1, 1);  // y variance
    var.robot(2) = state.sigma(2, 2);  // theta variance

    // Extract landmark variances
    var.landmarks.resize(NUM_LANDMARKS);
    for (size_t i = 0; i < NUM_LANDMARKS; ++i) {
      int idx             = landmarkIndex(i);
      var.landmarks[i](0) = state.sigma(idx, idx);  // landmark x variance
      var.landmarks[i](1) = state.sigma(idx + 1, idx + 1);  // landmark y
                                                            // variance
    }

    return var;
  }
};
