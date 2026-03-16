
/*
 * This file is part of the RACK (Robotics Application Construction Kit)
 * core project.
 *
 * Copyright (C) 2005-2008 STILL GmbH / Advanced Developement
 * All rights reserved.
 *
 */

#ifndef MAIN_COTRANS_H_
#define MAIN_COTRANS_H_

#include "position2d.h"

#define GSL_SUCCESS 1
#define GSL_ERROR   -1
/**
 * @{
 */
typedef float transMatrix2d[3][3];  ///< homogeneous 3x3 matrix

class CoTrans {
 public:
  CoTrans();
  ~CoTrans();

  static void getTransMatrix2d(transMatrix2d tm, const position_2d* off);
  static void getOffset(position_2d* off, transMatrix2d tm);

  static void multTransMatrix2d(transMatrix2d tm, transMatrix2d tm1,
                                transMatrix2d tm2);
  static void invertTransMatrix2d(transMatrix2d tm, transMatrix2d tmInv);
  static void transPosition2d(transMatrix2d tm, position_2d* p,
                              position_2d* pTrans);
};

#endif  // MAIN_COTRANS_H_

/**
 * @}
 */
