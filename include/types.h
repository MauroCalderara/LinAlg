/** \file
 *
 *  \brief            Type definitions and preprocessor macros.
 *
 *  \date             Created:  Jan 5, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_TYPES_H_
#define LINALG_TYPES_H_

#include <complex>

/** \def              fortran_name(x,y)
 *
 *  \brief            Fortran naming convention macro. Change names as per the
 *                    requirement of the linker.
 */
#ifndef fortran_name
#ifdef NoChange
#define fortran_name(x,y) (x)
#elif UpCase
#define fortran_name(x,y) (y)
#else
#define fortran_name(x,y) (x ## _)
#endif
#endif

namespace LinAlg {

/// Index type used in LinAlg::. Also update the MPI macro accordingly
typedef int I_t;
/// The integer type to use for MPI. Set to 'MPI_UNSIGNED' when using
/// 'unsigned int' or 'size_t' for LinAlg::I_t.
#define LINALG_MPI_INT MPI_INT

/// Definition of LAPACK single precision floating point, real
typedef float S_t;
/// Definition of LAPACK double precision floating point, real
typedef double D_t;
/// Definition of LAPACK single precision floating point, complex
typedef std::complex<float> C_t;
/// Definition of LAPACK double precision floating point, complex
typedef std::complex<double> Z_t;

/// Enum of datatypes
enum class Type {
  U,      //< Unknown type
  S,      //< Single precision floating point, real
  D,      //< Double precision floating point, real
  C,      //< Single precision floating point, complex
  Z,      //< Double precision floating point, complex
  I,      //< Integer type
};

/** \brief            Return the LinAlg::Type member corresponding to the
 *                    template instanciation
 *
 *  \returns          Type::U
 */
template <typename T> inline Type type()      { return Type::U; };
/** \overload
 *
 *  \returns          Type::S
 */
template <>           inline Type type<S_t>() { return Type::S; };
/** \overload
 *
 *  \returns          Type::D
 */
template <>           inline Type type<D_t>() { return Type::D; };
/** \overload
 *
 *  \returns          Type::C
 */
template <>           inline Type type<C_t>() { return Type::C; };
/** \overload
 *
 *  \returns          Type::Z
 */
template <>           inline Type type<Z_t>() { return Type::Z; };
/** \overload
 *
 *  \returns          Type::I
 */
template <>           inline Type type<I_t>() { return Type::I; };

/** \brief              Type casts.
 *
 *                      Use as cast<desired_type>(real, imag).
 *
 *  \param[in]          real
 *                      Real part.
 *
 *  \param[in]          imag
 *                      OPTIONAL: imaginary part. Ignored for S_t and D_t
 *                      conversions. If omitted for C_t and Z_t, the imaginary
 *                      part is set to zero.
 *
 *  \returns            The number in the requested type.
 *
 *  \note               All type casts optionally take two arguments to avoid
 *                      having to branch for being complex in templated
 *                      functions.
 */
template <typename T, typename U, typename V>
inline T cast(U real, V imag) { T tmp(real, imag); return tmp; };
/** \overload
 *
 *  \param[in]          real
 *                      Real part.
 *
 *  \returns            The number as the requested type
 */
template <typename T, typename U>
inline T cast(U real) { T tmp(real); return tmp; };

// These catch also cast<S_t>(1, 2) due to automatic casting of int, float,
// double, etc.
/** \overload
 *
 *  \param[in]          real
 *                      Real part
 *
 *  \param[in]          imag
 *                      Imaginary part
 *
 *  \returns            The number as S_t (float)
 */
template <>
inline S_t cast<S_t>(S_t real, S_t imag) { S_t tmp(real); return tmp; };
/** \overload
 *
 *  \param[in]          real
 *                      Real part.
 *
 *  \param[in]          imag
 *                      Imaginary part.
 *
 *  \returns            The number as D_t (double)
 */
template <>
inline D_t cast<D_t>(D_t real, D_t imag) { D_t tmp(real); return tmp; };


/** \brief            Storage locations
 */
enum class Location {
    host,       //< Main memory
#ifdef HAVE_CUDA
    GPU,        //< GPGPU
#endif
#ifdef HAVE_MIC
    MIC,        //< Intel Xeon Phi / MIC
#endif
#ifdef HAVE_MPI
    remote,     //< On a remote host
#endif
};

/** \brief            Storage formats
 */
enum class Format {
    ColMajor,   //< Column major (Fortran layout: [a_00, a_10, a_20, ...])
    RowMajor,   //< Row major (C/C++ layout: [a_00, a_01, a_02, ...])
    CSR,        //< Compressed Sparse Row
    CSC,        //< Compressed Sparse Column
};

/** \brief            Buffer types
 */
enum class BufferType {
  OnePass,      //< One pass in one direction, can start at either end
  TwoPass,      //< One pass in each direction, can start at either end
};

/** \brief            Buffer directions
 */
enum class BufferDirection {
  increasing,   //< Buffer runs in direction of increasing indices
  decreasing,   //< Buffer runs in direction of decreasing indices
};


} /* namespace LinAlg */

#ifdef HAVE_MKL
//#define MKL_INT LinAlg::I_t
/// Definition of single precision complex number within MKL
#define MKL_Complex8 LinAlg::C_t
/// Definition of double precision complex number within MKL
#define MKL_Complex16 LinAlg::Z_t
#include <mkl.h>
#endif

#endif /* LINALG_TYPES_H_ */
