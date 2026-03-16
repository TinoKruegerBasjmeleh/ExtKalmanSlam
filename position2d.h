/*
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
 * Authors
 *      Joerg Langenberg <joerg.langenberg@gmx.net>
 *
 */
#ifndef MAIN_DEFINES_POSITION2D_H_
#define MAIN_DEFINES_POSITION2D_H_

// ######################################################################
// # Position2D (static size - no message )
// ######################################################################

/**
 * position 2d structure
 * @ingroup main_defines
 */
typedef struct {
  float x{};   /**< [mm] x-coordinate */
  float y{};   /**< [mm] y-coordinate */
  float rho{}; /**< [rad] rotation about the z-axis, clockwise positive */
} position_2d;

#endif  // MAIN_DEFINES_POSITION2D_H_
