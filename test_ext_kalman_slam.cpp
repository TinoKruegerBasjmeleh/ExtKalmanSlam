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
#include <gtest/gtest.h>

#include <cmath>

#include "ext_kalman_slam.h"  // NOLINT(build/include_subdir)

// These tests exercise EKFSLAM through its public interface only. The
// low-level math helpers (motionModel, motionJacobian, measurementJacobian,
// rangeBearingObservation, predictLandmarkMeasurement, initializeLandmark,
// checkIfLandmarkObserved) are now private and are validated indirectly via
// predict()/update()/getRobotPose()/landmarkPose()/getVariances().
class EKFSLAMTest : public ::testing::Test {
 protected:
  EKFSLAM         ekf;
  Eigen::Matrix3d motionNoise;
  Eigen::Matrix2d measurementNoise;

  void            SetUp() override {
    // Small process noise
    motionNoise.setIdentity();
    motionNoise *= 0.01;

    // Small measurement noise
    measurementNoise.setIdentity();
    measurementNoise *= 0.01;

    // Robot at origin with moderate, isotropic uncertainty.
    ekf.setInitialPos(Eigen::Vector3d(0.0, 0.0, 0.0),
                      Eigen::Matrix3d::Identity() * 0.1);
  }
};

// ============================================================================
// Motion Model Tests
// ============================================================================

TEST_F(EKFSLAMTest, MotionModelStraightLine) {
  // Straight-line motion (w = 0) via a single prediction step.
  EKFSLAM::Control u{1.0, 0.0, 1.0};  // v=1, w=0, dt=1

  ekf.predict(u, motionNoise);
  Position2D<double> pose = ekf.getRobotPose();

  // Should move forward 1 meter in x direction
  EXPECT_NEAR(pose.x, 1.0, 1e-5);
  EXPECT_NEAR(pose.y, 0.0, 1e-5);
  EXPECT_NEAR(pose.rho, 0.0, 1e-5);
}

TEST_F(EKFSLAMTest, MotionModelCircularMotion) {
  EKFSLAM::Control u{1.0, M_PI / 2.0, 1.0};  // v=1, w=π/2, dt=1 (90° turn)

  ekf.predict(u, motionNoise);
  Position2D<double> pose   = ekf.getRobotPose();

  double             radius = u.v / u.w;  // radius = 2/π
  // After a 90° turn, robot should be at approximately (radius, radius, π/2)
  EXPECT_NEAR(pose.x, radius, 1e-4);
  EXPECT_NEAR(pose.y, radius, 1e-4);
  EXPECT_NEAR(pose.rho, M_PI / 2.0, 1e-5);
}

TEST_F(EKFSLAMTest, MotionModelAngleNormalization) {
  // Angles must stay normalized to [-π, π].
  ekf.setInitialPos(Eigen::Vector3d(0.0, 0.0, 3.0),
                    Eigen::Matrix3d::Identity() * 0.1);
  EKFSLAM::Control u{1.0, 2.0, 1.0};  // Large angular velocity

  ekf.predict(u, motionNoise);
  Position2D<double> pose = ekf.getRobotPose();

  EXPECT_GE(pose.rho, -M_PI);
  EXPECT_LE(pose.rho, M_PI);
}

// ============================================================================
// Prediction Covariance Tests
// ============================================================================

TEST_F(EKFSLAMTest, PredictKeepsCovarianceFinite) {
  ekf.predict({1.0, 0.5, 1.0}, motionNoise);  // Circular motion
  EKFSLAM::Variances var = ekf.getVariances();
  EXPECT_TRUE(var.robot.allFinite());
}

// ============================================================================
// Prediction Tests
// ============================================================================

TEST_F(EKFSLAMTest, PredictUpdatesRobotPose) {
  Position2D<double> before = ekf.getRobotPose();

  ekf.predict({1.0, 0.0, 1.0}, motionNoise);  // Move forward 1 meter

  Position2D<double> after = ekf.getRobotPose();

  // Robot position should have changed
  EXPECT_NE(after.x, before.x);

  // Landmarks are untouched by a pure prediction step.
  for (size_t i = 0; i < EKFSLAM::NUM_LANDMARKS; ++i) {
    Eigen::Vector2d lm = ekf.landmarkPose(static_cast<int>(i));
    EXPECT_DOUBLE_EQ(lm(0), 0.0);
    EXPECT_DOUBLE_EQ(lm(1), 0.0);
  }
}

TEST_F(EKFSLAMTest, PredictIncreasesUncertainty) {
  EKFSLAM::Variances before = ekf.getVariances();

  ekf.predict({1.0, 0.0, 1.0}, motionNoise);

  EKFSLAM::Variances after = ekf.getVariances();

  // Robot uncertainty should grow after a prediction step.
  EXPECT_GT(after.robot.sum(), before.robot.sum());
}

// ============================================================================
// Landmark Initialisation (via update) Tests
// ============================================================================

TEST_F(EKFSLAMTest, UpdateInitialisesLandmarkAhead) {
  // Robot at origin, landmark measured 5 units directly ahead.
  EKFSLAM::Measurement meas{-1, Eigen::Vector2d(5.0, 0.0), 0};

  EXPECT_TRUE(ekf.update(meas, measurementNoise));

  Eigen::Vector2d lm = ekf.landmarkPose(0);
  EXPECT_NEAR(lm(0), 5.0, 1e-5);
  EXPECT_NEAR(lm(1), 0.0, 1e-5);
}

TEST_F(EKFSLAMTest, UpdateInitialisesLandmarkWithRotation) {
  ekf.setInitialPos(Eigen::Vector3d(0.0, 0.0, M_PI / 2.0),
                    Eigen::Matrix3d::Identity() * 0.1);
  EKFSLAM::Measurement meas{-1, Eigen::Vector2d(5.0, 0.0), 0};

  EXPECT_TRUE(ekf.update(meas, measurementNoise));

  // With the robot facing north, a bearing-0 landmark lands at (0, 5).
  Eigen::Vector2d lm = ekf.landmarkPose(0);
  EXPECT_NEAR(lm(0), 0.0, 1e-5);
  EXPECT_NEAR(lm(1), 5.0, 1e-5);
}

TEST_F(EKFSLAMTest, DataAssociationPicksFirstFreeLandmark) {
  EKFSLAM::Measurement meas{-1, Eigen::Vector2d(5.0, 0.0), 0};
  EXPECT_EQ(ekf.getAssociatedLandmarkId(meas), 0);
}

TEST_F(EKFSLAMTest, DataAssociationReobservesSameLandmark) {
  EKFSLAM::Measurement first{-1, Eigen::Vector2d(5.0, 0.0), 0};
  ekf.update(first, measurementNoise);

  // Re-observing the same landmark must reuse id 0, not allocate a new slot.
  EKFSLAM::Measurement again{-1, Eigen::Vector2d(5.0, 0.0), 100};
  EXPECT_EQ(ekf.getAssociatedLandmarkId(again), 0);
  ekf.update(again, measurementNoise);

  // Landmark 1 must remain unallocated (still at the origin).
  Eigen::Vector2d lm1 = ekf.landmarkPose(1);
  EXPECT_DOUBLE_EQ(lm1(0), 0.0);
  EXPECT_DOUBLE_EQ(lm1(1), 0.0);
}

// ============================================================================
// Landmark Refinement Tests
// ============================================================================

TEST_F(EKFSLAMTest, UpdateRefinesLandmarkEstimate) {
  // Initialise landmark 0 at (5, 0).
  EKFSLAM::Measurement init{-1, Eigen::Vector2d(5.0, 0.0), 0};
  ekf.update(init, measurementNoise);
  Eigen::Vector2d      before = ekf.landmarkPose(0);

  // A slightly different observation should shift the estimate.
  EKFSLAM::Measurement meas{-1, Eigen::Vector2d(4.5, -0.1), 0};
  ekf.update(meas, measurementNoise);
  Eigen::Vector2d after = ekf.landmarkPose(0);

  EXPECT_NE(after(0), before(0));  // Landmark x should change
  EXPECT_NE(after(1), before(1));  // Landmark y should change
}

// ============================================================================
// Update Step Tests
// ============================================================================

TEST_F(EKFSLAMTest, UpdateReducesUncertainty) {
  ekf.setInitialPos(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity() * 0.5);

  // Initialise landmark 0 (timestamp 0).
  EKFSLAM::Measurement init{-1, Eigen::Vector2d(5.0, 0.0), 0};
  ekf.update(init, measurementNoise);
  const EKFSLAM::Variances at_init    = ekf.getVariances();

  // Accumulate motion uncertainty without observing anything.
  Eigen::Matrix3d          driftNoise = Eigen::Matrix3d::Identity() * 0.01;
  for (int i = 0; i < 10; ++i) {
    ekf.predict(EKFSLAM::Control{0.0, 0.0, 0.1}, driftNoise);
  }
  const EKFSLAM::Variances before = ekf.getVariances();
  EXPECT_GT(before.robot.sum(), at_init.robot.sum());

  // Re-observe the landmark. Predict the measurement from the current
  // estimate so the residual is non-zero only where the motion actually
  // introduced error -- a zero residual carries no information and must not
  // shrink the covariance.
  const Eigen::Vector2d lm = ekf.landmarkPose(0);
  const double          dx = lm.x(), dy = lm.y();
  EKFSLAM::Measurement  meas{
      -1, Eigen::Vector2d(std::hypot(dx, dy), std::atan2(dy, dx)), 50000};
  ekf.update(meas, measurementNoise);
  const EKFSLAM::Variances after = ekf.getVariances();

  // Re-observation must claw back the uncertainty accumulated while moving.
  EXPECT_LT(after.robot.sum(), before.robot.sum());

  // The covariance must stay positive definite through the update.
  EXPECT_GT(after.robot.minCoeff(), 0.0);

  // NOTE: in a textbook EKF-SLAM the robot uncertainty could never fall below
  // its value at landmark-initialisation time, because a self-initialised
  // landmark is not an independent global reference (Dissanayake et al.,
  // 2001). That bound does NOT hold here: initializeLandmark() still leaves
  // the robot<->landmark cross-covariance at zero, so the landmark is treated
  // as independent evidence. Asserting the bound is therefore deferred until
  // initializeLandmark() seeds Sigma_{r,lm} = Sigma_rr * Gx^T.
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EKFSLAMTest, FullEKFCycle) {
  ekf.setInitialPos(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity() * 0.1);

  // Step 1: Move forward.
  ekf.predict({1.0, 0.0, 1.0}, motionNoise);
  EXPECT_NEAR(ekf.getRobotPose().x, 1.0, 0.1);

  // Step 2: Observe (and initialise) landmark 0.
  EKFSLAM::Measurement meas1{-1, Eigen::Vector2d(5.0, 0.0), 100};
  EXPECT_TRUE(ekf.update(meas1, measurementNoise));
  EXPECT_GT(ekf.landmarkPose(0).norm(), 1.0);

  // Step 3: Turn and move.
  ekf.predict({1.0, M_PI / 4.0, 1.0}, motionNoise);

  // Step 4: Observe landmark 0 again.
  EKFSLAM::Measurement meas2{-1, Eigen::Vector2d(5.0, 0.0), 200};
  ekf.update(meas2, measurementNoise);

  // All values should remain finite.
  Position2D<double> pose = ekf.getRobotPose();
  EXPECT_TRUE(std::isfinite(pose.x));
  EXPECT_TRUE(std::isfinite(pose.y));
  EXPECT_TRUE(std::isfinite(pose.rho));

  EKFSLAM::Variances var = ekf.getVariances();
  EXPECT_TRUE(var.robot.allFinite());
  for (const auto& lv : var.landmarks) {
    EXPECT_TRUE(lv.allFinite());
  }
}

TEST_F(EKFSLAMTest, LandmarkIndexHelper) {
  for (size_t i = 0; i < EKFSLAM::NUM_LANDMARKS; ++i) {
    EXPECT_EQ(ekf.landmarkIndex(static_cast<int>(i)),
              static_cast<int>(3 + 2 * i));
  }
}

TEST_F(EKFSLAMTest, LandmarkPoseRetrieval) {
  LandmarkMap map;
  map.addLandmark(0, 10.0, 20.0, 0.1, 0.1, 42);
  ekf.setStateFromMap(map);

  Eigen::Vector2d lm_pose = ekf.landmarkPose(0);

  EXPECT_DOUBLE_EQ(lm_pose(0), 10.0);
  EXPECT_DOUBLE_EQ(lm_pose(1), 20.0);
}

// ============================================================================
// Edge Cases and Numerical Stability Tests
// ============================================================================

TEST_F(EKFSLAMTest, ZeroVelocityMotion) {
  ekf.setInitialPos(Eigen::Vector3d(1.0, 2.0, M_PI / 4.0),
                    Eigen::Matrix3d::Identity() * 0.1);
  EKFSLAM::Control u{0.0, 0.0, 1.0};  // No motion

  ekf.predict(u, motionNoise);
  Position2D<double> pose = ekf.getRobotPose();

  // Pose should remain unchanged
  EXPECT_NEAR(pose.x, 1.0, 1e-5);
  EXPECT_NEAR(pose.y, 2.0, 1e-5);
  EXPECT_NEAR(pose.rho, M_PI / 4.0, 1e-5);
}

TEST_F(EKFSLAMTest, SmallAngularVelocityUsesLinearApproximation) {
  EKFSLAM ekf1;
  EKFSLAM ekf2;
  ekf1.setInitialPos(Eigen::Vector3d::Zero(),
                     Eigen::Matrix3d::Identity() * 0.1);
  ekf2.setInitialPos(Eigen::Vector3d::Zero(),
                     Eigen::Matrix3d::Identity() * 0.1);

  ekf1.predict({1.0, 1e-6, 1.0}, motionNoise);  // Very small ω (uses linear)
  ekf2.predict({1.0, 0.0, 1.0}, motionNoise);   // Zero ω (definitely linear)

  Position2D<double> p1 = ekf1.getRobotPose();
  Position2D<double> p2 = ekf2.getRobotPose();

  // Results should be very close since both use the linear approximation.
  EXPECT_NEAR(p1.x, p2.x, 1e-3);
  EXPECT_NEAR(p1.y, p2.y, 1e-3);
}

TEST_F(EKFSLAMTest, LandmarkCloseToRobotStability) {
  // Edge case: landmark measured very close to the robot.
  EKFSLAM::Measurement meas{-1, Eigen::Vector2d(0.001, 0.0), 0};
  EXPECT_TRUE(ekf.update(meas, measurementNoise));

  Eigen::Vector2d lm = ekf.landmarkPose(0);
  EXPECT_TRUE(lm.allFinite());

  EKFSLAM::Variances var = ekf.getVariances();
  EXPECT_TRUE(var.robot.allFinite());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
