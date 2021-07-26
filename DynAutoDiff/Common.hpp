#ifndef __DYNAUTODIFF_COMMON__
#define __DYNAUTODIFF_COMMON__
#include <numbers>

namespace DynAutoDiff {
enum Reduction { None = 0, Sum = 1, Mean = 2 };
enum Dim { Row = 0, Col = 1, All = -1 };
}; // namespace DynAutoDiff
#endif