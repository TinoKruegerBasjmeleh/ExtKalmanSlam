#ifndef MAIN_COTRANS_H_
#define MAIN_COTRANS_H_

#include <array>
#include <cmath>
#include <cstddef>

// Header-only 2D coordinate transforms, templated on scalar type T.
// Instantiate with `double` for full precision or `float` for compactness.

template <typename T>
struct Position2D {
  T x{};
  T y{};
  T rho{};
};

template <typename T>
using TransMatrix2D = std::array<std::array<T, 3>, 3>;

template <typename T>
class CoTransT {
 public:
  static void getTransMatrix2d(TransMatrix2D<T>& tm, const Position2D<T>& off) {
    const T c = std::cos(off.rho);
    const T s = std::sin(off.rho);

    tm[0][0] = c;     tm[0][1] = -s;    tm[0][2] = off.x;
    tm[1][0] = s;     tm[1][1] = c;     tm[1][2] = off.y;
    tm[2][0] = T(0);  tm[2][1] = T(0);  tm[2][2] = T(1);
  }

  static void getOffset(Position2D<T>& off, const TransMatrix2D<T>& tm) {
    off.x   = tm[0][2];
    off.y   = tm[1][2];
    off.rho = std::atan2(-tm[0][1], tm[0][0]);
  }

  static void multTransMatrix2d(TransMatrix2D<T>&       tm,
                                const TransMatrix2D<T>& tm1,
                                const TransMatrix2D<T>& tm2) {
    TransMatrix2D<T> out{};
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        T acc = T(0);
        for (std::size_t k = 0; k < 3; ++k) {
          acc += tm1[i][k] * tm2[k][j];
        }
        out[i][j] = acc;
      }
    }
    tm = out;
  }

  static void invertTransMatrix2d(TransMatrix2D<T>&       tmInv,
                                  const TransMatrix2D<T>& tm) {
    tmInv[0][0] = tm[0][0];
    tmInv[0][1] = tm[1][0];
    tmInv[1][0] = tm[0][1];
    tmInv[1][1] = tm[1][1];

    tmInv[0][2] = -(tm[0][0] * tm[0][2] + tm[1][0] * tm[1][2]);
    tmInv[1][2] = -(tm[0][1] * tm[0][2] + tm[1][1] * tm[1][2]);

    tmInv[2][0] = T(0);
    tmInv[2][1] = T(0);
    tmInv[2][2] = T(1);
  }

  static void transPosition2d(const TransMatrix2D<T>& tm,
                              const Position2D<T>&    p,
                              Position2D<T>&          pTrans) {
    TransMatrix2D<T> tmP{};
    TransMatrix2D<T> tmPTrans{};
    getTransMatrix2d(tmP, p);
    multTransMatrix2d(tmPTrans, tm, tmP);
    getOffset(pTrans, tmPTrans);
  }
};

#endif  // MAIN_COTRANS_H_
