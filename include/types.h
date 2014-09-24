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

// For an explanation, see below under "TYPE DISCUSSION"
#ifdef HAVE_MAGMA
#include <magma_types.h>
#else
#include <complex>
#endif /* HAVE_MAGMA */

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

/* TYPE DISCUSSION
 *
 * Getting the types right for the various combinations of libraries turns out 
 * to be a somewhat subtle and messy issue. The approach used in LinAlg is 
 * this:
 *
 * If support for MAGMA is enabled (-DHAVE_MAGMA), we use MAGMA's complex 
 * types for our complex types. MAGMA in turn either uses what CUDA or MKL 
 * uses.
 * Otherwise we use the C++ complex types from <complex>.
 *
 * For either case functions like real() and imag() are provided in the LinAlg 
 * namespace.
 */
#ifdef MAGMA_TYPES_H

  /// Index type used in LinAlg::. Also update the MPI macro accordingly when 
  /// changing it.
  typedef magma_int_t I_t;

  // Here we try to replicate what's in magma_types.h:
  #if defined(MAGMA_ILP64) || defined(MKL_ILP64)
    /// The integer type to use for MPI.
    #define LINALG_MPI_INT MPI_LONG_LONG
  #else
    /// The integer type to use for MPI.
    #define LINALG_MPI_INT MPI_INT
  #endif

  /// Definition of LAPACK single precision floating point, real
  typedef float S_t;
  /// Definition of LAPACK double precision floating point, real
  typedef double D_t;
  /// Definition of LAPACK single precision floating point, complex
  typedef magmaFloatComplex C_t;
  /// Definition of LAPACK double precision floating point, complex
  typedef magmaDoubleComplex Z_t;

  /** \brief          Return real part of complex number
   *
   *  \param          z
   *                  Complex number.
   *
   *  \returns        real part.
   */
  template <>
  inline S_t real<C_t>(C_t z) { return MAGMA_C_REAL(z); };
  /** \overload */
  template <>
  inline S_t imag<C_t>(C_t z) { return MAGMA_C_IMAG(z); };
  /** \overload */
  template <>
  inline D_t real<Z_t>(Z_t z) { return MAGMA_Z_REAL(z); };
  /** \overload */
  template <>
  inline D_t imag<Z_t>(Z_t z) { return MAGMA_Z_IMAG(z); };

#else /* MAGMA_TYPES_H is not defined */

  /// Index type used in LinAlg::. Also update the MPI macro accordingly when 
  /// changing it.
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

  /** \brief          Return real part of complex number
   *
   *  \param          z
   *                  Complex number.
   *
   *  \returns        real part.
   */
  template <>
  inline S_t real<C_t>(C_t z) { return std::real(z); };
  /** \overload */
  template <>
  inline S_t imag<C_t>(C_t z) { return std::imag(z); };
  /** \overload */
  template <>
  inline D_t real<Z_t>(Z_t z) { return std::real(z); };
  /** \overload */
  template <>
  inline D_t imag<Z_t>(Z_t z) { return std::imag(z); };

#endif /* MAGMA_TYPES_H */

/// Enum of datatypes
enum class Type {
  O,      //< Other type
  S,      //< Single precision floating point, real
  D,      //< Double precision floating point, real
  C,      //< Single precision floating point, complex
  Z,      //< Double precision floating point, complex
  I,      //< Integer type
};

/** \brief            Return the LinAlg::Type member corresponding to the
 *                    template instanciation
 *
 *  \returns          Type::O
 */
template <typename T> inline Type type()      { return Type::O; };
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
 *
 *  \note             This enum only includes members for which there is
 *                    compiled support. Thus, if you want to check for a 
 *                    certain location in a portable code you either have to 
 *                    use #ifdefs or use the .is_on_X() members of Dense<T> 
 *                    and Sparse<T>
 */
enum class Location {
    host,       //< Main memory
#ifdef HAVE_CUDA
    GPU,        //< GPGPU
#endif
#ifdef HAVE_MIC
    MIC,        //< Intel Xeon Phi / MIC
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
