/** \file
 *
 *  \brief            Forward declaration of utility functions required in
 *                    dense.h and sparse.h
 */

#ifndef LINALG_UTILITIES_FORWARD_H_
#define LINALG_UTILITIES_FORWARD_H_

#include <string>
#include <tuple>

#include "../types.h"
#include <memory>

namespace LinAlg {

template <typename T>
struct Dense;

template <typename T>
struct Sparse;

namespace Utilities {

template <typename T>
inline std::shared_ptr<T> host_make_shared(I_t);

template <typename T>
void copy_1Darray(T*, I_t, T*, Location, int, Location, int);

template <typename T>
void copy_2Darray(bool, Format, const T*, I_t, Location, int, I_t, I_t, Format,
                  T*, I_t, Location, int);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Dense<U>&, SubBlock, Location, int);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Dense<U>&, Location, int);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Dense<U>&);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Sparse<U>&, SubBlock, Location, int);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Sparse<U>&, Location, int);

template <typename T, typename U>
void reallocate_like(Dense<T>&, const Sparse<U>&);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Dense<U>&, SubBlock, Location, int);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Dense<U>&, Location, int);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Dense<U>&);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Sparse<U>&, SubBlock, Location, int);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Sparse<U>&, Location, int);

template <typename T, typename U>
void reallocate_like(Sparse<T>&, const Sparse<U>&);

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_FORWARD_H_ */
