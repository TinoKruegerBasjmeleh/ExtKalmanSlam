/*!
 * RACK - Robotics Application Construction Kit
 * Copyright (C) 2005-2006 University of Hannover
 *                         Institute for Systems Engineering - RTS
 *                         Professor Bernardo Wagner
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * @brief This library provides basic angle functions for floating point radian
 * angles
 * @author Joerg Langenberg <joerg.langenberg@gmx.net>
 * @ingroup library
 * @defgroup angle-tool AngleTool
 *
 * @{
 */

#ifndef MAIN_ANGLE_TOOL_H_
#define MAIN_ANGLE_TOOL_H_

#include <math.h>

#include <iostream>
#include <type_traits>

class AngleTool {
 public:
  /****************************************************************************/
  /*!
   * @brief normalise angle between 0 and 2 pi
   *
   * normalise a floating point type angle in radians such that the output angle
   * is between 0 and 2 pi
   * @param angle angle to normalise
   * @return the normalised angle
   ****************************************************************************/

  template <typename T>
  static T normaliseAngle(T angle) {
    static_assert(std::is_floating_point<T>::value, "Floating point required.");

    if (angle < 0.0) {
      return normaliseAngle(angle + 2.0 * M_PI);
    }

    if (angle >= 2.0 * M_PI) {
      return normaliseAngle(angle - 2.0 * M_PI);
    }

    return angle;
  }

  /****************************************************************************/
  /*!
   * @brief normalise angle between -pi and pi
   *
   * normalise a floating point type angle in radians such that the output angle
   * is between -pi and pi
   * @param angle angle to normalise
   * @return the normalised angle
   ****************************************************************************/

  template <typename T>
  static T normaliseAngleSym0(T angle) {
    static_assert(std::is_floating_point<T>::value, "Floating point required.");
    if (angle <= -M_PI) {
      return normaliseAngleSym0(angle + 2.0 * M_PI);
    }

    if (angle > M_PI) {
      return normaliseAngleSym0(angle - 2.0 * M_PI);
    }

    return angle;
  }

  /****************************************************************************/
  /*!
   * @brief calculate a difference between two angles
   *
   * calculate the difference between angleB and angleA (angleB - angleA)
   * @param angleB the minuend angle
   * @param angleA the subtrahend angle to subtract from angleB
   * @return the difference between "angleB" and "angleA"
   *
   ****************************************************************************/

  template <typename T>
  static T deltaAngle(T angleA, T angleB) {
    static_assert(std::is_floating_point<T>::value, "Floating point required.");
    angleB = normaliseAngleSym0(angleB);
    angleA = normaliseAngleSym0(angleA);
    return normaliseAngleSym0(angleB - angleA);
  }

  /****************************************************************************/
  /*!
   * @brief  differenceAngle
   * calculate the smallest difference
   * between phi1 and phi2 including the signs from
   * floating point type angles.
   * The angles will be normalized to -PI...PI
   * @param  phi1                       - minuend angle
   * @param  phi2                       - subtrahend angle
   * @return the offset
   ****************************************************************************/

  template <typename T>
  static T differenceAngle(T phi1, T phi2) {
    static_assert(std::is_floating_point<T>::value, "Floating point required.");
    T   tmp_phi, norm_phi1, norm_phi2;
    int sign  = 1;

    // norm to 0 ... 2PI:
    // -------------------
    norm_phi1 = normaliseAngle(phi1);
    norm_phi2 = normaliseAngle(phi2);

    sign      = (norm_phi1 < norm_phi2) ? (-1) : (1);
    // get difference:
    // ---------------
    if (norm_phi2 > norm_phi1) {
      tmp_phi = norm_phi2 - norm_phi1;
    } else {
      tmp_phi = norm_phi1 - norm_phi2;
    }

    // test which difference is the smallest

    tmp_phi = std::min(tmp_phi, static_cast<T>(2 * M_PI - tmp_phi));

    return sign * normaliseAngleSym0(tmp_phi);
  }
};
/*!
 * @}
 */

#endif  // MAIN_ANGLE_TOOL_H_
