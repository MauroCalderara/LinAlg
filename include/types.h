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

#ifndef LINALG_NO_CHECK
#include <cstdio>
#endif

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

  /// Construction of S_t
  #define LINALG_MAKE_Ct(r, i) MAGMA_C_MAKE(r, i)
  /// Construction of Z_t
  #define LINALG_MAKE_Zt(r, i) MAGMA_Z_MAKE(r, i)

  /** \brief          Return real part of a number
   *
   *  \param          z
   *                  Number.
   *
   *  \returns        Real part.
   */
  inline S_t real(S_t z) { return z; };
  /** \overload */
  inline D_t real(D_t z) { return z; };
  /** \overload */
  inline S_t real(C_t z) { return MAGMA_C_REAL(z); };
  /** \overload */
  inline D_t real(Z_t z) { return MAGMA_Z_REAL(z); };

  /** \brief          Return imaginary part of a number
   *
   *  \param          z
   *                  Number.
   *
   *  \returns        Imaginary part.
   */
  inline S_t imag(S_t z) { return 0.0; };
  /** \overload */
  inline D_t imag(D_t z) { return 0.0; };
  /** \overload */
  inline S_t imag(C_t z) { return MAGMA_C_IMAG(z); };
  /** \overload */
  inline D_t imag(Z_t z) { return MAGMA_Z_IMAG(z); };

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
  
  /// Construction of S_t
  #define LINALG_MAKE_Ct(r, i) C_t(r, i)
  /// Construction of Z_t
  #define LINALG_MAKE_Zt(r, i) Z_t(r, i)

  /** \brief          Return real part of a number
   *
   *  \param          z
   *                  Number.
   *
   *  \returns        Real part.
   */
  inline S_t real(S_t z) { return z; };
  /** \overload */
  inline D_t real(D_t z) { return z; };
  /** \overload */
  inline S_t real(C_t z) { return std::real(z); };
  /** \overload */
  inline D_t real(Z_t z) { return std::real(z); };

  /** \brief          Return imaginary part of a number
   *
   *  \param          z
   *                  Number.
   *
   *  \returns        Imaginary part.
   */
  inline S_t imag(S_t z) { return 0.0; };
  /** \overload */
  inline D_t imag(D_t z) { return 0.0; };
  /** \overload */
  inline S_t imag(C_t z) { return std::imag(z); };
  /** \overload */
  inline D_t imag(Z_t z) { return std::imag(z); };

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


/////////////
// Type casts
/** \brief              Cast from two numbers
 *
 *                      Use as cast<desired_type>(real, imag).
 *
 *  \param[in]          r
 *                      Real part.
 *
 *  \param[in]          i
 *                      Imaginary part.
 *
 *  \returns            The number in the requested type.
 */
template <typename T, typename U, typename V>
#ifndef LINALG_NO_CHECK
// If this exception is thrown, there is a specialization missing below 
// (probably you tried to convert from two complex numbers to a real number).
inline T cast(U r, V i) {
  std::printf("Error in %s:%d\n", __FILE__, __LINE__);
  std::printf("   (Most likely you tried to convert an integer type using "
             "    cast<>(), which is not supported.\n");
  throw;
  return 0;
};
#else
inline T cast(U r, V i) { return 0; };
#endif
#ifndef DOXYGEN_SKIP
// This sucks: we basically specialize all possible combinations of casts so 
// why is this 'templated' anyway? The reason is that this way we can use the 
// cast<T> syntax within templated code.
template <> inline S_t cast<S_t>(S_t r, S_t i) { return r; };
template <> inline S_t cast<S_t>(S_t r, D_t i) { return r; };
template <> inline S_t cast<S_t>(D_t r, S_t i) { return S_t(r); };
template <> inline S_t cast<S_t>(D_t r, D_t i) { return S_t(r); };

template <> inline D_t cast<D_t>(S_t r, S_t i) { return D_t(r); };
template <> inline D_t cast<D_t>(S_t r, D_t i) { return D_t(r); };
template <> inline D_t cast<D_t>(D_t r, S_t i) { return r; };
template <> inline D_t cast<D_t>(D_t r, D_t i) { return r; };

template <> inline C_t cast<C_t>(S_t r, S_t i) { 
  return LINALG_MAKE_Ct(r, i); 
};
template <> inline C_t cast<C_t>(S_t r, D_t i) { 
  return LINALG_MAKE_Ct(r, S_t(i));
};
template <> inline C_t cast<C_t>(D_t r, S_t i) {
  return LINALG_MAKE_Ct(S_t(r), i);
};
template <> inline C_t cast<C_t>(D_t r, D_t i) {
  return LINALG_MAKE_Ct(S_t(r), S_t(i));
};

template <> inline Z_t cast<Z_t>(S_t r, S_t i) { 
  return LINALG_MAKE_Zt(D_t(r), D_t(i)); 
};
template <> inline Z_t cast<Z_t>(S_t r, D_t i) { 
  return LINALG_MAKE_Zt(D_t(r), i);
};
template <> inline Z_t cast<Z_t>(D_t r, S_t i) {
  return LINALG_MAKE_Zt(r, D_t(i));
};
template <> inline Z_t cast<Z_t>(D_t r, D_t i) {
  return LINALG_MAKE_Zt(r, i);
};
#endif

/** \brief              Cast from single number
 *
 *  Use as cast<desired_type>(v);
 *
 *  \param[in]          v
 *                      Value.
 *
 *  \returns            The value as the requested type
 */
template <typename T, typename U>
#ifndef LINALG_NO_CHECKS
// If this exception is thrown, there is something strange since all 
// combinations should be specialized below ... 
inline T cast(U value) {
  std::printf("Error in %s:%d\n", __FILE__, __LINE__);
  std::printf("   (Most likely you tried to convert an integer type using "
             "    cast<>(), which is not supported.\n");
  throw;
  return 0;
};
#else
inline T cast(U value) { return 0; };
#endif
#ifndef DOXYGEN_SKIP
// See above for an explanation of this
template <> inline S_t cast<S_t>(S_t v) { return v; };
template <> inline S_t cast<S_t>(D_t v) { return S_t(v); };
template <> inline S_t cast<S_t>(C_t v) { return S_t(real(v)); };
template <> inline S_t cast<S_t>(Z_t v) { return S_t(real(v)); };

template <> inline D_t cast<D_t>(S_t v) { return D_t(v); };
template <> inline D_t cast<D_t>(D_t v) { return v; };
template <> inline D_t cast<D_t>(C_t v) { return D_t(real(v)); };
template <> inline D_t cast<D_t>(Z_t v) { return D_t(real(v)); };

template <> inline C_t cast<C_t>(S_t v) { return LINALG_MAKE_Ct(v, 0); };
template <> inline C_t cast<C_t>(D_t v) { return LINALG_MAKE_Ct(D_t(v), 0); };
template <> inline C_t cast<C_t>(C_t v) { return v; };
template <> inline C_t cast<C_t>(Z_t v) { 
  return LINALG_MAKE_Ct(S_t(real(v)), S_t(imag(v)));
};

template <> inline Z_t cast<Z_t>(S_t v) { return LINALG_MAKE_Zt(v, 0); };
template <> inline Z_t cast<Z_t>(D_t v) { return LINALG_MAKE_Zt(D_t(v), 0); };
template <> inline Z_t cast<Z_t>(C_t v) { 
  return LINALG_MAKE_Zt(D_t(real(v)), D_t(imag(v)));
};
template <> inline Z_t cast<Z_t>(Z_t v) { return v; };
#endif

/** \brief            Matrix properties
 *
 *  We store matrix properties as bitfield internally. The matrices' setter 
 *  members check for internal consistency.
 */
enum Property {
  General   = 0x01,
  Symmetric = 0x02,
  Hermitian = 0x04,
  Packed    = 0x08,
};


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
