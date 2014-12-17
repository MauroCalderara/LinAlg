/** \file
 *
 *  \brief            Forward declarations required in dense.h and sparse.h
 */

#ifndef LINALG_FORWARD_H_
#define LINALG_FORWARD_H_

#include <string>
#include <tuple>

#include "types.h"
#include <memory>

namespace LinAlg {

// dense.h
template <typename T>
struct Dense;

// sparse.h
template <typename T>
struct Sparse;

// copy.h
template <typename T>
inline void copy(const Dense<T>&, Dense<T>&);

template <typename T>
inline void copy(const Dense<T>&, Sparse<T>&);

template <typename T>
inline void copy(const Sparse<T>&, Dense<T>&);

template <typename T>
inline void copy(const Sparse<T>&, Sparse<T>&);

template <typename T>
inline void copy(const Dense<T>&, SubBlock, Dense<T>&);

template <typename T>
inline void copy(const Dense<T>&, SubBlock, Sparse<T>&);

template <typename T>
inline void copy(const Sparse<T>&, SubBlock, Dense<T>&);

template <typename T>
inline void copy(const Sparse<T>&, SubBlock, Sparse<T>&);

} /* namespace LinAlg */

#include "utilities/utilities_forward.h"

#endif /* LINALG_FORWARD_H_ */
