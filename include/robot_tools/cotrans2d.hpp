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
 *         Dennis Schüthe <dennis_schuethe@kiongroup.com>
 *         Volker Viereck <volker.viereck@kiongroup.com>
 *         Felix Hess     <felix.hess@kiongroup.com>
 *         Tino Krueger-Basjmeleh <tino.krueger@still.de.de>
 ******************************************************************************/
#ifndef ROBOT_TOOLS__COTRANS2D_HPP_
#define ROBOT_TOOLS__COTRANS2D_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

// Header-only 2D counterpart of robot_tools/cotrans.hpp (CoTrans). Same API shape:
// static template members, scalar T deduced from TransMatrix2D<T>, and any struct
// (or pointer to it) exposing x, y [, rho] as pose/point. Kept C++17 (no concepts).

template <typename T>
struct Pose2D
{
  T x{};
  T y{};
  T rho{};
};

template <typename T>
using TransMatrix2D = std::array<std::array<T, 3>, 3>;  // homogeneous 3x3 matrix

namespace cotrans2d_detail
{
template <typename U, typename = void>
struct HasXY : std::false_type
{
};
template <typename U>
struct HasXY<U, std::void_t<decltype(std::declval<U &>().x), decltype(std::declval<U &>().y)>>
: std::true_type
{
};

template <typename U, typename = void>
struct HasRho : std::false_type
{
};
template <typename U>
struct HasRho<U, std::void_t<decltype(std::declval<U &>().rho)>> : std::true_type
{
};

template <typename U>
constexpr decltype(auto) Deref(U & u)
{
  if constexpr (std::is_pointer_v<std::remove_cv_t<U>>) {
    return *u;
  } else {
    return u;
  }
}

template <typename U>
using Pointee = std::remove_cv_t<std::remove_reference_t<decltype(Deref(std::declval<U &>()))>>;

template <typename U>
constexpr bool kIsLvalueRefOrPointer =
  std::is_lvalue_reference_v<U> || std::is_pointer_v<std::remove_reference_t<U>>;
}  // namespace cotrans2d_detail

class CoTrans2D
{
public:
  /**
   * @brief Get the transformation matrix from a given offset.
   * @param[out] tm The transformation matrix to fill.
   * @param[in] off Struct (or pointer to it) with x, y and optional rho (0 if absent).
   */
  template <typename T, typename U>
  static void GetTransMatrix(TransMatrix2D<T> & tm, const U & off)
  {
    static_assert(std::is_floating_point_v<T>, "TransMatrix2D scalar must be floating point");
    using V = cotrans2d_detail::Pointee<const U>;
    static_assert(cotrans2d_detail::HasXY<V>::value, "GetTransMatrix needs x and y");
    const auto & o = cotrans2d_detail::Deref(off);

    T rho{};
    if constexpr (cotrans2d_detail::HasRho<V>::value) {
      rho = static_cast<T>(o.rho);
    }
    const T c = std::cos(rho);
    const T s = std::sin(rho);

    tm[0][0] = c;
    tm[0][1] = -s;
    tm[0][2] = static_cast<T>(o.x);
    tm[1][0] = s;
    tm[1][1] = c;
    tm[1][2] = static_cast<T>(o.y);
    tm[2][0] = T(0);
    tm[2][1] = T(0);
    tm[2][2] = T(1);
  }

  /**
   * @brief Invert the given transformation matrix.
   * @param[in] tm The transformation matrix to invert.
   * @param[out] tmInv The inverted transformation matrix.
   */
  template <typename T>
  static void InvertTransMatrix(const TransMatrix2D<T> & tm, TransMatrix2D<T> & tmInv)
  {
    const TransMatrix2D<T> in = tm;  // allow tm and tmInv to alias

    tmInv[0][0] = in[0][0];
    tmInv[0][1] = in[1][0];
    tmInv[1][0] = in[0][1];
    tmInv[1][1] = in[1][1];

    tmInv[0][2] = -(in[0][0] * in[0][2] + in[1][0] * in[1][2]);
    tmInv[1][2] = -(in[0][1] * in[0][2] + in[1][1] * in[1][2]);

    tmInv[2][0] = T(0);
    tmInv[2][1] = T(0);
    tmInv[2][2] = T(1);
  }

  /**
   * @brief Multiply two transformation matrices.
   * @param tm[out] The resulting transformation matrix (may alias tm1 / tm2).
   * @param tm1[in] The first transformation matrix.
   * @param tm2[in] The second transformation matrix.
   */
  template <typename T>
  static void MultTransMatrix(
    TransMatrix2D<T> & tm, const TransMatrix2D<T> & tm1, const TransMatrix2D<T> & tm2)
  {
    TransMatrix2D<T> out{};
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 3; ++k) {
          out[i][j] += tm1[i][k] * tm2[k][j];
        }
      }
    }
    tm = out;
  }

  /**
   * @brief Get the offset from the given TM.
   * @param off[out] Struct (or pointer to it) with x, y, rho to fill.
   * @param tm[in] The transformation matrix.
   */
  template <typename T, typename U>
  static void GetOffset(U && off, const TransMatrix2D<T> & tm)
  {
    static_assert(cotrans2d_detail::kIsLvalueRefOrPointer<U>, "GetOffset needs lvalue or pointer");
    using V = cotrans2d_detail::Pointee<U>;
    static_assert(
      cotrans2d_detail::HasXY<V>::value && cotrans2d_detail::HasRho<V>::value,
      "GetOffset needs x, y and rho");
    auto & o = cotrans2d_detail::Deref(off);
    using lat_data_t = std::remove_reference_t<decltype(o.x)>;
    using rot_data_t = std::remove_reference_t<decltype(o.rho)>;

    o.x = static_cast<lat_data_t>(tm[0][2]);
    o.y = static_cast<lat_data_t>(tm[1][2]);
    o.rho = static_cast<rot_data_t>(std::atan2(tm[1][0], tm[0][0]));
  }

  /**
   * @brief Transform the given point with the given TM (all other fields are copied).
   * @param[in] tm The transformation matrix.
   * @param[in] p The point to transform.
   * @param[out] p_trans The transformed point.
   */
  template <typename T, typename U, typename W>
  static void TransPoint(const TransMatrix2D<T> & tm, const U & p, W && p_trans)
  {
    static_assert(cotrans2d_detail::kIsLvalueRefOrPointer<W>, "TransPoint needs lvalue or pointer");
    static_assert(
      std::is_same_v<cotrans2d_detail::Pointee<const U>, cotrans2d_detail::Pointee<W>>,
      "TransPoint input and output must have the same type");
    static_assert(
      cotrans2d_detail::HasXY<cotrans2d_detail::Pointee<W>>::value, "TransPoint needs x and y");
    const auto in = cotrans2d_detail::Deref(p);  // copy: allow p and p_trans to alias
    auto & out = cotrans2d_detail::Deref(p_trans);
    using lat_data_t = std::remove_reference_t<decltype(out.x)>;

    out = in;
    out.x = static_cast<lat_data_t>(
      tm[0][0] * static_cast<T>(in.x) + tm[0][1] * static_cast<T>(in.y) + tm[0][2]);
    out.y = static_cast<lat_data_t>(
      tm[1][0] * static_cast<T>(in.x) + tm[1][1] * static_cast<T>(in.y) + tm[1][2]);
  }

  /**
   * @brief Transform the given position (x, y, rho) with the given TM.
   * @param[in] tm The transformation matrix.
   * @param[in] p The position to transform.
   * @param[out] p_trans The transformed position.
   */
  template <typename T, typename U, typename W>
  static void TransPosition(const TransMatrix2D<T> & tm, const U & p, W && p_trans)
  {
    TransMatrix2D<T> tm_point{};
    TransMatrix2D<T> tm_point_trans{};

    GetTransMatrix(tm_point, p);
    MultTransMatrix(tm_point_trans, tm, tm_point);
    GetOffset(std::forward<W>(p_trans), tm_point_trans);
  }
};

#endif  // ROBOT_TOOLS__COTRANS2D_HPP_
