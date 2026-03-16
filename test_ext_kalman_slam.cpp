#include "ext_kalman_slam.h"

#include <gtest/gtest.h>

#include <cmath>

class EKFSLAMTest : public ::testing::Test {
 protected:
  EKFSLAM              ekf;
  EKFSLAM::EKFState    state;
  Eigen::Matrix3d      motionNoise;
  Eigen::Matrix2d      measurementNoise;

  void SetUp() override {
    // Initialize state with robot at origin and two landmarks
    state.mu    = Eigen::VectorXd::Zero(7);  // [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y]
    state.sigma = Eigen::MatrixXd::Identity(7, 7) * 0.1;

    // Small process noise
    motionNoise.setIdentity();
    motionNoise *= 0.01;

    // Small measurement noise
    measurementNoise.setIdentity();
    measurementNoise *= 0.01;
  }
};

// ============================================================================
// Motion Model Tests
// ============================================================================

TEST_F(EKFSLAMTest, MotionModelStraightLine) {
  // Test straight-line motion (w ≈ 0)
  Eigen::Vector3d     pose(0.0, 0.0, 0.0);
  EKFSLAM::Control    u{1.0, 0.0, 1.0};  // v=1, w=0, dt=1

  Eigen::Vector3d     new_pose = ekf.motionModel(pose, u);

  // Should move forward 1 meter in x direction
  EXPECT_NEAR(new_pose(0), 1.0, 1e-5);
  EXPECT_NEAR(new_pose(1), 0.0, 1e-5);
  EXPECT_NEAR(new_pose(2), 0.0, 1e-5);
}

TEST_F(EKFSLAMTest, MotionModelCircularMotion) {
  // Test circular motion
  Eigen::Vector3d  pose(0.0, 0.0, 0.0);
  EKFSLAM::Control u{1.0, M_PI / 2.0, 1.0};  // v=1, w=π/2, dt=1 (90° turn)

  Eigen::Vector3d  new_pose = ekf.motionModel(pose, u);

  double           radius   = u.v / u.w;  // radius = 2/π
  // After 90° turn, robot should be at approximately (radius, radius, π/2)
  EXPECT_NEAR(new_pose(0), radius, 1e-4);
  EXPECT_NEAR(new_pose(1), radius, 1e-4);
  EXPECT_NEAR(new_pose(2), M_PI / 2.0, 1e-5);
}

TEST_F(EKFSLAMTest, MotionModelAngleNormalization) {
  // Test that angles are normalized to [-π, π]
  Eigen::Vector3d  pose(0.0, 0.0, 3.0);  // theta = 3 radians
  EKFSLAM::Control u{1.0, 2.0, 1.0};     // Large angular velocity

  Eigen::Vector3d  new_pose = ekf.motionModel(pose, u);

  // Check that angle is in valid range
  EXPECT_GE(new_pose(2), -M_PI);
  EXPECT_LE(new_pose(2), M_PI);
}

// ============================================================================
// Motion Jacobian Tests
// ============================================================================

TEST_F(EKFSLAMTest, MotionJacobianStraightLine) {
  Eigen::Vector3d  pose(0.0, 0.0, M_PI / 4.0);  // 45° orientation
  EKFSLAM::Control u{1.0, 0.0, 1.0};            // Straight motion

  Eigen::Matrix3d  G = ekf.motionJacobian(pose, u);

  // Check diagonal elements should be 1
  EXPECT_NEAR(G(0, 0), 1.0, 1e-5);
  EXPECT_NEAR(G(1, 1), 1.0, 1e-5);
  EXPECT_NEAR(G(2, 2), 1.0, 1e-5);

  // Check theta derivatives
  double           expected_dx_dtheta = -u.v * u.dt * std::sin(pose(2));
  double           expected_dy_dtheta = u.v * u.dt * std::cos(pose(2));
  EXPECT_NEAR(G(0, 2), expected_dx_dtheta, 1e-5);
  EXPECT_NEAR(G(1, 2), expected_dy_dtheta, 1e-5);
}

TEST_F(EKFSLAMTest, MotionJacobianCircularMotion) {
  Eigen::Vector3d  pose(0.0, 0.0, 0.0);
  EKFSLAM::Control u{1.0, 0.5, 1.0};  // Circular motion

  Eigen::Matrix3d  G = ekf.motionJacobian(pose, u);

  // Jacobian should be well-formed
  EXPECT_TRUE(G.allFinite());

  // Diagonal should still be 1
  EXPECT_NEAR(G(0, 0), 1.0, 1e-5);
  EXPECT_NEAR(G(1, 1), 1.0, 1e-5);
  EXPECT_NEAR(G(2, 2), 1.0, 1e-5);
}

// ============================================================================
// Prediction Tests
// ============================================================================

TEST_F(EKFSLAMTest, PredictUpdatesRobotPose) {
  state.mu(0)          = 0.0;
  state.mu(1)          = 0.0;
  state.mu(2)          = 0.0;

  EKFSLAM::Control u   = {1.0, 0.0, 1.0};  // Move forward 1 meter

  Eigen::VectorXd  mu_before = state.mu;

  ekf.predict(state, u, motionNoise);

  // Robot position should have changed
  EXPECT_NE(state.mu(0), mu_before(0));

  // Landmarks should remain unchanged
  EXPECT_DOUBLE_EQ(state.mu(3), mu_before(3));
  EXPECT_DOUBLE_EQ(state.mu(4), mu_before(4));
  EXPECT_DOUBLE_EQ(state.mu(5), mu_before(5));
  EXPECT_DOUBLE_EQ(state.mu(6), mu_before(6));
}

TEST_F(EKFSLAMTest, PredictIncreasesUncertainty) {
  Eigen::MatrixXd sigma_before = state.sigma;

  EKFSLAM::Control u           = {1.0, 0.0, 1.0};

  ekf.predict(state, u, motionNoise);

  // Covariance should increase (robot uncertainty grows)
  double trace_before = sigma_before.trace();
  double trace_after  = state.sigma.trace();
  EXPECT_GT(trace_after, trace_before);
}

// ============================================================================
// Range-Bearing Observation Tests
// ============================================================================

TEST_F(EKFSLAMTest, RangeBearingObservationAtOrigin) {
  state.mu(0) = 0.0;  // x
  state.mu(1) = 0.0;  // y
  state.mu(2) = 0.0;  // theta

  // Measurement: range=5, bearing=0 (directly ahead)
  Eigen::Vector2d z(5.0, 0.0);

  Eigen::Vector2d landmark_pos = ekf.rangeBearingObservation(state.mu, z);

  // Landmark should be at (5, 0)
  EXPECT_NEAR(landmark_pos(0), 5.0, 1e-5);
  EXPECT_NEAR(landmark_pos(1), 0.0, 1e-5);
}

TEST_F(EKFSLAMTest, RangeBearingObservationWithRotation) {
  state.mu(0) = 0.0;           // x
  state.mu(1) = 0.0;           // y
  state.mu(2) = M_PI / 2.0;    // theta = 90°

  // Measurement: range=5, bearing=0 (relative to robot heading)
  Eigen::Vector2d z(5.0, 0.0);

  Eigen::Vector2d landmark_pos = ekf.rangeBearingObservation(state.mu, z);

  // With robot facing north, landmark at bearing 0 should be at (0, 5)
  EXPECT_NEAR(landmark_pos(0), 0.0, 1e-5);
  EXPECT_NEAR(landmark_pos(1), 5.0, 1e-5);
}

// ============================================================================
// Landmark Measurement Prediction Tests
// ============================================================================

TEST_F(EKFSLAMTest, PredictLandmarkMeasurementSimple) {
  // Robot at origin
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;

  // Landmark 0 at (5, 0)
  state.mu(3) = 5.0;
  state.mu(4) = 0.0;

  Eigen::Vector2d z_hat = ekf.predictLandmarkMeasurement(state.mu, 0);

  // Should predict: range=5, bearing=0
  EXPECT_NEAR(z_hat(0), 5.0, 1e-5);  // range
  EXPECT_NEAR(z_hat(1), 0.0, 1e-5);  // bearing
}

TEST_F(EKFSLAMTest, PredictLandmarkMeasurementWithOffset) {
  // Robot at (1, 1, 0)
  state.mu(0) = 1.0;
  state.mu(1) = 1.0;
  state.mu(2) = 0.0;

  // Landmark 0 at (4, 5)
  state.mu(3) = 4.0;
  state.mu(4) = 5.0;

  Eigen::Vector2d z_hat = ekf.predictLandmarkMeasurement(state.mu, 0);

  // dx = 3, dy = 4, range = 5
  EXPECT_NEAR(z_hat(0), 5.0, 1e-5);

  // bearing = atan2(4, 3) ≈ 0.927 radians
  double expected_bearing = std::atan2(4.0, 3.0);
  EXPECT_NEAR(z_hat(1), expected_bearing, 1e-5);
}

// ============================================================================
// Measurement Jacobian Tests
// ============================================================================

TEST_F(EKFSLAMTest, MeasurementJacobianDimensions) {
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;
  state.mu(3) = 5.0;
  state.mu(4) = 5.0;

  Eigen::Matrix<double, 2, 5> H = ekf.measurementJacobian(state.mu, 0);

  // Check dimensions
  EXPECT_EQ(H.rows(), 2);
  EXPECT_EQ(H.cols(), 5);

  // Check that values are finite
  EXPECT_TRUE(H.allFinite());
}

TEST_F(EKFSLAMTest, MeasurementJacobianValues) {
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;
  state.mu(3) = 3.0;
  state.mu(4) = 4.0;

  Eigen::Matrix<double, 2, 5> H = ekf.measurementJacobian(state.mu, 0);

  // For landmark at (3, 4) from origin: range=5, bearing=atan2(4,3)
  double dx     = 3.0;
  double dy     = 4.0;
  double q      = dx * dx + dy * dy;  // 25
  double sqrt_q = std::sqrt(q);       // 5

  // Check range derivatives
  EXPECT_NEAR(H(0, 0), -dx / sqrt_q, 1e-5);  // ∂r/∂x = -3/5
  EXPECT_NEAR(H(0, 1), -dy / sqrt_q, 1e-5);  // ∂r/∂y = -4/5
  EXPECT_NEAR(H(0, 2), 0.0, 1e-5);           // ∂r/∂θ = 0

  // Check bearing derivatives
  EXPECT_NEAR(H(1, 0), dy / q, 1e-5);   // ∂φ/∂x = 4/25
  EXPECT_NEAR(H(1, 1), -dx / q, 1e-5);  // ∂φ/∂y = -3/25
  EXPECT_NEAR(H(1, 2), -1.0, 1e-5);     // ∂φ/∂θ = -1
}

// ============================================================================
// Landmark Initialization Tests
// ============================================================================

TEST_F(EKFSLAMTest, CheckIfLandmarkObserved) {
  // Initially, landmarks are at (0, 0) - not observed
  EXPECT_FALSE(ekf.checkIfLandmarkObserved(state, 0));
  EXPECT_FALSE(ekf.checkIfLandmarkObserved(state, 1));

  // Set landmark 0 position
  state.mu(3) = 5.0;
  state.mu(4) = 3.0;

  EXPECT_TRUE(ekf.checkIfLandmarkObserved(state, 0));
  EXPECT_FALSE(ekf.checkIfLandmarkObserved(state, 1));
}

TEST_F(EKFSLAMTest, InitializeLandmark) {
  // Robot at origin
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;

  // Measurement: range=10, bearing=π/4
  Eigen::Vector2d z(10.0, M_PI / 4.0);

  // Measurement noise
  Eigen::Matrix2d Q = Eigen::Matrix2d::Identity() * 0.1;

  ekf.initializeLandmark(state, 0, z, Q);

  // Landmark should be initialized at correct global position
  double expected_x = 10.0 * std::cos(M_PI / 4.0);
  double expected_y = 10.0 * std::sin(M_PI / 4.0);

  EXPECT_NEAR(state.mu(3), expected_x, 1e-5);
  EXPECT_NEAR(state.mu(4), expected_y, 1e-5);
}

// ============================================================================
// Update Step Tests
// ============================================================================

TEST_F(EKFSLAMTest, UpdateInitializesNewLandmark) {
  // Robot at origin
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;

  EKFSLAM::Measurement meas{0, Eigen::Vector2d(5.0, 0.0)};

  ekf.update(state, meas, measurementNoise);

  // Landmark should be initialized
  EXPECT_TRUE(ekf.checkIfLandmarkObserved(state, 0));
  EXPECT_NEAR(state.mu(3), 5.0, 1e-5);
  EXPECT_NEAR(state.mu(4), 0.0, 1e-5);
}

TEST_F(EKFSLAMTest, UpdateRefinesLandmarkEstimate) {
  // Initialize robot and landmark with some uncertainty
  state.mu(0)    = 0.0;
  state.mu(1)    = 0.0;
  state.mu(2)    = 0.0;
  state.mu(3)    = 5.1;  // Landmark with small error
  state.mu(4)    = 0.1;
  state.sigma   *= 1.0;  // Higher initial uncertainty

  // Perfect measurement
  EKFSLAM::Measurement meas{0, Eigen::Vector2d(5.0, 0.0)};

  Eigen::VectorXd mu_before = state.mu;

  ekf.update(state, meas, measurementNoise);

  // State should have been corrected toward measurement
  EXPECT_NE(state.mu(3), mu_before(3));  // Landmark x should change
  EXPECT_NE(state.mu(4), mu_before(4));  // Landmark y should change
}

TEST_F(EKFSLAMTest, UpdateReducesUncertainty) {
  // Initialize landmark
  state.mu(3) = 5.0;
  state.mu(4) = 0.0;

  Eigen::MatrixXd sigma_before = state.sigma;

  // Get a measurement
  EKFSLAM::Measurement meas{0, Eigen::Vector2d(5.0, 0.0)};

  ekf.update(state, meas, measurementNoise);

  // Robot uncertainty should decrease
  double robot_variance_before = sigma_before.block<3, 3>(0, 0).trace();
  double robot_variance_after  = state.sigma.block<3, 3>(0, 0).trace();
  EXPECT_LT(robot_variance_after, robot_variance_before);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EKFSLAMTest, FullEKFCycle) {
  // Start at origin
  state.mu.setZero();
  state.sigma = Eigen::MatrixXd::Identity(7, 7) * 0.1;

  // Step 1: Move forward
  EKFSLAM::Control u1{1.0, 0.0, 1.0};
  ekf.predict(state, u1, motionNoise);

  // Robot should be around (1, 0, 0)
  EXPECT_NEAR(state.mu(0), 1.0, 0.1);

  // Step 2: Observe landmark 0
  EKFSLAM::Measurement meas1{0, Eigen::Vector2d(5.0, 0.0)};
  ekf.update(state, meas1, measurementNoise);

  // Landmark 0 should be initialized
  EXPECT_TRUE(ekf.checkIfLandmarkObserved(state, 0));

  // Step 3: Turn and move
  EKFSLAM::Control u2{1.0, M_PI / 4.0, 1.0};
  ekf.predict(state, u2, motionNoise);

  // Step 4: Observe landmark 0 again
  Eigen::Vector2d z_hat = ekf.predictLandmarkMeasurement(state.mu, 0);
  EKFSLAM::Measurement meas2{0, z_hat};  // Use predicted measurement
  ekf.update(state, meas2, measurementNoise);

  // All values should be finite
  EXPECT_TRUE(state.mu.allFinite());
  EXPECT_TRUE(state.sigma.allFinite());

  // Covariance should be positive semi-definite
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(state.sigma);
  EXPECT_GE(es.eigenvalues().minCoeff(), -1e-10);  // Allow small numerical errors
}

TEST_F(EKFSLAMTest, LandmarkIndexHelper) {
  EXPECT_EQ(ekf.landmarkIndex(0), 3);
  EXPECT_EQ(ekf.landmarkIndex(1), 5);
}

TEST_F(EKFSLAMTest, LandmarkPoseRetrieval) {
  state.mu(3) = 10.0;
  state.mu(4) = 20.0;

  Eigen::Vector2d lm_pose = ekf.landmarkPose(state, 0);

  EXPECT_DOUBLE_EQ(lm_pose(0), 10.0);
  EXPECT_DOUBLE_EQ(lm_pose(1), 20.0);
}

// ============================================================================
// Edge Cases and Numerical Stability Tests
// ============================================================================

TEST_F(EKFSLAMTest, ZeroVelocityMotion) {
  Eigen::Vector3d  pose(1.0, 2.0, M_PI / 4.0);
  EKFSLAM::Control u{0.0, 0.0, 1.0};  // No motion

  Eigen::Vector3d  new_pose = ekf.motionModel(pose, u);

  // Pose should remain unchanged
  EXPECT_NEAR(new_pose(0), pose(0), 1e-5);
  EXPECT_NEAR(new_pose(1), pose(1), 1e-5);
  EXPECT_NEAR(new_pose(2), pose(2), 1e-5);
}

TEST_F(EKFSLAMTest, SmallAngularVelocityUsesLinearApproximation) {
  Eigen::Vector3d  pose(0.0, 0.0, 0.0);
  EKFSLAM::Control u1{1.0, 1e-6, 1.0};   // Very small ω (should use linear)
  EKFSLAM::Control u2{1.0, 0.0, 1.0};    // Zero ω (definitely linear)

  Eigen::Vector3d  result1 = ekf.motionModel(pose, u1);
  Eigen::Vector3d  result2 = ekf.motionModel(pose, u2);

  // Results should be very close since both use linear approximation
  EXPECT_NEAR(result1(0), result2(0), 1e-3);
  EXPECT_NEAR(result1(1), result2(1), 1e-3);
}

TEST_F(EKFSLAMTest, LandmarkAtRobotPositionStability) {
  // Edge case: landmark very close to robot
  state.mu(0) = 0.0;
  state.mu(1) = 0.0;
  state.mu(2) = 0.0;
  state.mu(3) = 0.001;
  state.mu(4) = 0.001;

  // Should still compute valid Jacobian
  Eigen::Matrix<double, 2, 5> H = ekf.measurementJacobian(state.mu, 0);
  EXPECT_TRUE(H.allFinite());

  // Should still predict measurement
  Eigen::Vector2d z_hat = ekf.predictLandmarkMeasurement(state.mu, 0);
  EXPECT_TRUE(z_hat.allFinite());
  EXPECT_GT(z_hat(0), 0.0);  // Range should be positive
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
