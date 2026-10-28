/******************************************************************************
 * Copyright (C) 2005-2006 University of Hannover
 *                         Institute for Systems Engineering - RTS
 *                         Professor Bernardo Wagner
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Authors
 *         Joerg Langenberg <joerg.langenberg@gmx.net>
 *         Tino Krueger-Basjmeleh <tino.krueger@still.de.de>
 ******************************************************************************/
#ifndef ROBOT_TOOLS__ANGLETOOL_HPP_
#define ROBOT_TOOLS__ANGLETOOL_HPP_

#include <math.h>

#include <iostream>
#include <type_traits>

class AngleTool
{
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
  static T normaliseAngle(T angle)
  {
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
  static T normaliseAngleSym0(T angle)
  {
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
  static T deltaAngle(T angleA, T angleB)
  {
    static_assert(std::is_floating_point<T>::value, "Floating point required.");
    angleB = normaliseAngleSym0(angleB);
    angleA = normaliseAngleSym0(angleA);
    return normaliseAngleSym0(angleB - angleA);
  }
};
/*!
 * @}
 */
#endif  // ROBOT_TOOLS__ANGLETOOL_HPP_
