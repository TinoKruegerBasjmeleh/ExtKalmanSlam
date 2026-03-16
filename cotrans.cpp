/*!
 * This file is part of the RACK (Robotics Application Construction Kit)
 * core project.
 *
 * Copyright (C) 2005-2008 STILL GmbH / Advanced Developement
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 *@author Volker Viereck    <volker.viereck@still.de>
 *@author Carsten Schulz    <carsten.schulz@still.de>
 *@author Tino Krüger       <tino.krueger@still.de>
 *
 * @brief cotrans offers all coordinate and point transformations in a 2D
 *environment.
 *
 * The methods are related to the formulas of John J. Craig, "Introduction to
 *Robotics" 3rd edition, pages 41 to 43. It is used the X-Y-Z fixed angles
 *notation. See also:
 *
 *
 * Convention in reference to the book:
 *   Code   =   Book
 * @f$ \phi = \gamma @f$
 * @f$ \psi = \beta @f$
 * @f$ \rho = \alpha @f$
 *
 *
 * @defgroup cotrans CoTrans
 * @ingroup library
 * @{
 */

#include "cotrans.h"
#include <cmath>
#include <cstring>

CoTrans::CoTrans() {}

CoTrans::~CoTrans() {}

/*****************************************************************************/
/**
 *  \brief  Get transformation matrix to given Offset related to the original
 *coordinate system \param  tm  - computed transformation matrix \param  off -
 *corresponding Offset
 *****************************************************************************/
void CoTrans::getTransMatrix2d(transMatrix2d tm, const position_2d* off) {
  float cos_phi, sin_phi;

  cos_phi = cos(off->rho);  // rho in [rad]
  sin_phi = sin(off->rho);

  memset(tm, 0, sizeof(transMatrix2d));
  tm[0][0] = cos_phi;
  tm[0][1] = -sin_phi;
  tm[0][2] = (float)off->x;
  tm[1][0] = sin_phi;
  tm[1][1] = cos_phi;
  tm[1][2] = (float)off->y;
  tm[2][2] = 1;

  return;
}

/*****************************************************************************/
/**
 *  \brief  Invert a homogeneous 3x3 matrix
 *  \param  tm  - the transformation matrix
 *  \param tmInv return the inverted matrix
 *****************************************************************************/
void CoTrans::invertTransMatrix2d(transMatrix2d tm, transMatrix2d tmInv) {
  memcpy(tmInv, tm, sizeof(transMatrix2d));

  // orientation:
  // ------------
  tmInv[0][0] = tm[0][0];
  tmInv[0][1] = tm[1][0];
  tmInv[1][0] = tm[0][1];
  tmInv[1][1] = tm[1][1];

  // translation:
  // ------------
  tmInv[0][2] = -(tm[0][0] * tm[0][2] + tm[1][0] * tm[1][2]);
  tmInv[1][2] = -(tm[0][1] * tm[0][2] + tm[1][1] * tm[1][2]);

  // homogeneous extensions:
  // -----------------------
  tmInv[2][0] = 0;
  tmInv[2][1] = 0;
  tmInv[2][2] = 1;
}

/*****************************************************************************/
/**
 *  \brief  Multiply the given matrices in the way: tm = tm1 * tm2
 *  \param  tm1, tm2 - the 2D matrices to multiply
 *  \param  tm return the calculated matrix as pointer
 *****************************************************************************/
void CoTrans::multTransMatrix2d(transMatrix2d tm, transMatrix2d tm1,
                                transMatrix2d tm2) {
  int i, j, k;

  memset(tm, 0, sizeof(transMatrix2d));

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++) {
        tm[i][j] += tm1[i][k] * tm2[k][j];
      }
}

/*****************************************************************************/
/**
 *  \brief  Transform the given position with the given TM.
 *  \param  tm      - the transMatrix2d
 *  \param  p       - the position to transform
 *  \param  pTrans - the transformed position
 *****************************************************************************/
void CoTrans::transPosition2d(transMatrix2d tm, position_2d* p,
                              position_2d* pTrans) {
  transMatrix2d tmP, tmPTrans;

  getTransMatrix2d(tmP, p);
  multTransMatrix2d(tmPTrans, tm, tmP);
  getOffset(pTrans, tmPTrans);
}

/*****************************************************************************/
/**
 *  \brief  Get the corresponding offset to the given transMatrix2d
 *  \param  tm  - given transMatrix2d
 *  \param  off - the converted offset
 *  \return the offset
 *****************************************************************************/
void CoTrans::getOffset(position_2d* off, transMatrix2d tm) {
  // position:
  // ---------
  off->x   = (int)tm[0][2];
  off->y   = (int)tm[1][2];

  // orientation:
  // ------------
  off->rho = std::atan2(-tm[0][1], tm[0][0]);
}

/*!
 * @}
 */
