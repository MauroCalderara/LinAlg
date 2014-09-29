/** \file
 *
 *  \brief            xGETRI
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_GETRI_H_
#define LINALG_LAPACK_GETRI_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *        'Abstract' functions like 'solve' and 'invert'
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CUDA/cuda_checks.h"

#ifdef HAVE_MAGMA
#include <magma.h>
#endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */


#include "../types.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../dense.h"
#include "ilaenv.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(sgetri, SGETRI)(const I_t* n, S_t* A, const I_t* lda,
                                    const I_t* ipiv, S_t* work,
                                    const I_t* lwork, int* info);
  void fortran_name(dgetri, DGETRI)(const I_t* n, D_t* A, const I_t* lda,
                                    const I_t* ipiv, D_t* work,
                                    const I_t* lwork, int* info);
  void fortran_name(cgetri, CGETRI)(const I_t* n, C_t* A, const I_t* lda,
                                    const I_t* ipiv, C_t* work,
                                    const I_t* lwork, int* info);
  void fortran_name(zgetri, ZGETRI)(const I_t* n, Z_t* A, const I_t* lda,
                                    const I_t* ipiv, Z_t* work,
                                    const I_t* lwork, int* info);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            GETRI
 *
 *  P * L * U <- A^{-1}
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in]        work
 *
 *  \param[in]        lwork
 *
 *  \param[in,out]    info
 *
 * See [DGETRI](http://www.math.utah.edu/software/lapack/lapack-d/dgetri.html)
 */
inline void xGETRI(I_t n, S_t* A, I_t lda, int* ipiv, S_t* work, int lwork,
                   int* info) {
  fortran_name(sgetri, SGETRI)(&n, A, &lda, ipiv, work, &lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, D_t* A, I_t lda, int* ipiv, D_t* work, int lwork,
                   int* info) {
  fortran_name(dgetri, DGETRI)(&n, A, &lda, ipiv, work, &lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, C_t* A, I_t lda, int* ipiv, C_t* work, int lwork,
                   int* info) {
  fortran_name(cgetri, CGETRI)(&n, A, &lda, ipiv, work, &lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, Z_t* A, I_t lda, int* ipiv, Z_t* work, int lwork,
                   int* info) {
  fortran_name(zgetri, ZGETRI)(&n, A, &lda, ipiv, work, &lwork, info);
};

} /* namespace LinAlg::LAPACK::FORTRAN */

#ifdef HAVE_CUDA
namespace CUBLAS {

/** \brief            Invert a matrix in LU decomposed format
 *
 *  C <- A^(-1) = (P * L * U)^(-1)
 *
 *  NOTE: cublas' getri is out-of-place (the inverse is stored in C)
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in,out]    C
 *
 *  \param[in]        ldc
 *
 *  \param[in,out]    info
 *
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGETRI(cublasHandle_t handle, I_t n, S_t* A, I_t lda, int* ipiv,
                   S_t* C, I_t ldc, int* info) {
  checkCUBLAS(cublasSgetriBatched(handle, n, &A, lda, ipiv, &C, ldc, info, 1));
};
/** \overload
 */
inline void xGETRI(cublasHandle_t handle, I_t n, D_t* A, I_t lda, int* ipiv,
                   D_t* C, I_t ldc, int* info) {
  checkCUBLAS(cublasDgetriBatched(handle, n, &A, lda, ipiv, &C, ldc, info, 1));
};
/** \overload
 */
inline void xGETRI(cublasHandle_t handle, I_t n, C_t* A, I_t lda, int* ipiv,
                   C_t* C, I_t ldc, int* info) {
  checkCUBLAS(cublasCgetriBatched(handle, n, \
                                  reinterpret_cast<cuComplex**>(&A), lda, \
                                  ipiv, reinterpret_cast<cuComplex**>(&C), \
                                  ldc, info, 1));
};
/** \overload
 */
inline void xGETRI(cublasHandle_t handle, I_t n, Z_t* A, I_t lda, int* ipiv,
                   Z_t* C, I_t ldc, int* info) {
  checkCUBLAS(cublasZgetriBatched(handle, n, \
                                  reinterpret_cast<cuDoubleComplex**>(&A), lda,\
                                  ipiv, \
                                  reinterpret_cast<cuDoubleComplex**>(&C), \
                                  ldc, info, 1));
};

/** \brief            Invert multiple matrices in LU decomposed format
 *
 *  Carray[i] <- Aarray[i]^(-1) = (P * L * U)[i]^(-1)
 *
 *  NOTE: cublas' getri is out-of-place (the inverse is stored in C)
 *
 *  \param[in]        handle
 *
 *  \param[in]        n
 *
 *  \param[in,out]    Aarray
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    PivotArray
 *
 *  \param[in,out]    Carray
 *
 *  \param[in]        ldc
 *
 *  \param[in,out]    infoArray
 *
 *  \param[in]        batchSize
 *
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGETRI_batched(cublasHandle_t handle, I_t n, S_t* Aarray[], I_t lda,
                           int* PivotArray, S_t* Carray[], I_t ldc,
                           int* infoArray, I_t batchSize) {
  checkCUBLAS(cublasSgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, \
                                  ldc, infoArray, batchSize));
};
/** \overload
 */
inline void xGETRI_batched(cublasHandle_t handle, I_t n, D_t* Aarray[], I_t lda,
                           int* PivotArray, D_t* Carray[], I_t ldc,
                           int* infoArray, I_t batchSize) {
  checkCUBLAS(cublasDgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, \
                                  ldc, infoArray, batchSize));
};
/** \overload
 */
inline void xGETRI_batched(cublasHandle_t handle, I_t n, C_t* Aarray[], I_t lda,
                           int* PivotArray, C_t* Carray[], I_t ldc,
                           int* infoArray, I_t batchSize) {
  checkCUBLAS(cublasCgetriBatched(handle, n, \
                                  reinterpret_cast<cuComplex**>(Aarray), lda, \
                                  PivotArray, \
                                  reinterpret_cast<cuComplex**>(Carray), \
                                  ldc, infoArray, batchSize));
};
/** \overload
 */
inline void xGETRI_batched(cublasHandle_t handle, I_t n, Z_t* Aarray[], I_t lda,
                           int* PivotArray, Z_t* Carray[], I_t ldc,
                           int* infoArray, I_t batchSize) {
  checkCUBLAS(cublasZgetriBatched(handle, n, \
                                  reinterpret_cast<cuDoubleComplex**>(Aarray), \
                                  lda, PivotArray, \
                                  reinterpret_cast<cuDoubleComplex**>(Carray), \
                                  ldc, infoArray, batchSize));
};

} /* namespace LinAlg::LAPACK::CUBLAS */

#ifdef HAVE_MAGMA
namespace MAGMA {

/** \brief            GETRI
 *
 *  P * L * U <- A^{-1}
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in]        work
 *
 *  \param[in]        lwork
 *
 *  \param[in,out]    info
 *
 * See [DGETRI](http://www.math.utah.edu/software/lapack/lapack-d/dgetri.html) 
 * or the MAGMA sources
 */
inline void xGETRI(I_t n, S_t* A, I_t lda, int* ipiv, S_t* work, int lwork,
                   int* info) {
  magma_sgetri_gpu(n, A, lda, ipiv, work, lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, D_t* A, I_t lda, int* ipiv, D_t* work, int lwork,
                   int* info) {
  magma_dgetri_gpu(n, A, lda, ipiv, work, lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, C_t* A, I_t lda, int* ipiv, C_t* work, int lwork,
                   int* info) {
  magma_cgetri_gpu(n, A, lda, ipiv, work, lwork, info);
};
/** \overload
 */
inline void xGETRI(I_t n, Z_t* A, I_t lda, int* ipiv, Z_t* work, int lwork,
                   int* info) {
  magma_zgetri_gpu(n, A, lda, ipiv, work, lwork, info);
};

#ifndef DOXYGEN_SKIP
/*  Utility routines to determine the right size of lwork for xGETRI */
template <typename T>
inline I_t get_xgetri_nb(I_t n) {
  throw excBadArgument("get_xgetri_nb: must explicitly state the template to "
                       "use (e.g. get_xgetri_nb<D_t>(n))");
  return 0;
}
template <>
inline I_t get_xgetri_nb<S_t>(I_t n) { return magma_get_sgetri_nb(n); };
template <>
inline I_t get_xgetri_nb<D_t>(I_t n) { return magma_get_dgetri_nb(n); };
template <>
inline I_t get_xgetri_nb<C_t>(I_t n) { return magma_get_cgetri_nb(n); };
template <>
inline I_t get_xgetri_nb<Z_t>(I_t n) { return magma_get_zgetri_nb(n); };
#endif /* DOXYGEN_SKIP */

} /* namespace LinAlg::LAPACK::MAGMA */
#endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_input_transposed;

/** \brief            Compute the inverse using the LU decomposition of a
 *                    matrix in-place
 *
 *  Note that in-place inversion on GPU using CUBLAS matrices requires extra
 *  memory and an additional memory copy as the CUBLAS inversion function is
 *  out-of-place.
 *
 *  A = A**-1 (in-place)
 *
 *  \param[in]        A
 *                    Matrix in LU decomposition.
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). The CUBLAS backend
 *                    allows specifying an empty matrix for ipiv, assuming that
 *                    the input is a non-pivoting LU decomposition (see xGETRF).
 *
 *  \param[in]        work
 *                    For the matrices in main memory work is a vector of 
 *                    length at least A.rows(). The optimal length can be 
 *                    determined using LAPACK's ILAENV. Some backends (e.g.  
 *                    the CUBLAS backend) don't require a preallocated work in 
 *                    which case the supplied vector is ignored. If the 
 *                    backend does require a preallocated work and none or an 
 *                    empty one is specified, the routine will allocate one of 
 *                    the optimal size.
 */
template <typename T>
inline void xGETRI(Dense<T>& A, Dense<int>& ipiv, Dense<T>& work) {

#ifndef LINALG_NO_CHECKS
  // Check A
  check_input_transposed(A, "xGETRI(A, ipiv, work), A");
  if (A._rows != A._cols) {
    throw excBadArgument("xGETRI(A, ipiv, work), A: matrix must be square");
  }

  // Check if ipiv is empty _and_ we use the CUBLAS backend
  bool ipiv_empty;
#ifndef HAVE_CUDA
  ipiv_empty = false;
#else
  ipiv_empty = (ipiv._rows == 0) ? true : false;
#ifndef USE_MAGMA_GETRI
  if (ipiv_empty && A._location != Location::GPU) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: empty ipiv is only "
                         "allowed when inverting matrices on the GPU and using "
                         "the CUBLAS backend");
  }
#else
  if (ipiv_empty && A._location == Location::GPU) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: empty ipiv is not "
                         "supported when using the MAGMA backend");
  }
  if (ipiv._location != Location::host) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: ipiv must be allocated "
                         "in main memory for MAGMA backend");
  }
#endif /* USE_MAGMA_GETRI */
#endif /* HAVE_CUDA */

  if (!ipiv_empty) {

#ifndef USE_MAGMA_GETRI
    check_device(A, ipiv, "xGETRI()");
#endif
    check_input_transposed(ipiv, "xGETRI(A, ipiv, work), ipiv");
    if (A.rows() != ipiv.rows()) {
      throw excBadArgument("xGETRI(A, ipiv, work), ipiv: argument matrix size "
                           "mismatch: if ipiv is not empty, A.rows() = %d must "
                           "equal ipiv.rows() = %d", A.rows(), ipiv.rows());
    }
  
  }

#endif /* LINALG_NO_CHECKS */

  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto ipiv_ptr = ipiv._begin();
  int  info = 0;

  if (A._location == Location::host) {

    auto lwork = work._rows;

    // If work is empty, we have to allocate it optimally
    if (lwork == 0) {

      using LinAlg::Type;
      using LinAlg::LAPACK::FORTRAN::ILAENV;

      switch (type<T>()) {
        case Type::S:
          lwork = ILAENV(1, "sgetri", "", n, -1, -1, -1);
          break;
        case Type::D:
          lwork = ILAENV(1, "dgetri", "", n, -1, -1, -1);
          break;
        case Type::C:
          lwork = ILAENV(1, "cgetri", "", n, -1, -1, -1);
          break;
        case Type::Z:
          lwork = ILAENV(1, "zgetri", "", n, -1, -1, -1);
          break;
        default:
#ifndef LINALG_NO_CHECKS
          throw excBadArgument("xGETRI(): unsupported data type");
#endif /* LINALG_NO_CHECKS */
          break;
      }

      work.reallocate(lwork, 1);

    } 
#ifndef LINALG_NO_CHECKS
    else if (lwork < A._rows) {
    
      throw excBadArgument("xGETRI(A, ipiv, work), work: work must have at "
                           "least A.rows() = %d rows (work.rows() = %d)",
                           A._rows, lwork);
    }
#endif /* LINALG_NO_CHECKS */

    auto work_ptr = work._begin();
    FORTRAN::xGETRI(n, A_ptr, lda, ipiv_ptr, work_ptr, lwork, &info);

  }
#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

#ifndef USE_MAGMA_GETRI
    // CUBLAS' getri is out-of-place so we need to allocate a C and then stream
    // it back into A after the operation

    Dense<T> C(A._rows, A._cols, A._location, A._device_id);

    auto device_id = A._device_id;
    auto C_ptr = C._begin();
    auto ldc = C._leading_dimension;
    auto ipiv_ptr = (ipiv_empty) ? nullptr : ipiv._begin();

    using LinAlg::CUDA::CUBLAS::handles;
    CUBLAS::xGETRI(handles[device_id], n, A_ptr, lda, ipiv_ptr, C_ptr, ldc,
                   &info);

    A << C;

#else /* USE_MAGMA_GETRI */

    auto lwork = work._rows;

    // If work is empty, we have to allocate it optimally (in main memory!)
    if (lwork == 0) {

      using LinAlg::Type;
      using LinAlg::LAPACK::FORTRAN::ILAENV;

      lwork = MAGMA::get_xgetri_nb<T>(n) * n;

      work.reallocate(lwork, 1);

    } 
#ifndef LINALG_NO_CHECKS 
    else if (lwork < A._rows) {
      throw excBadArgument("xGETRI(A, ipiv, work), work: work must have at "
                           "least A.rows() = %d rows (work.rows() = %d)",
                           A._rows, lwork);
    }
#endif /* LINALG_NO_CHECKS */

    auto work_ptr = work._begin();
    MAGMA::xGETRI(n, A_ptr, lda, ipiv_ptr, work_ptr, lwork, &info);

#endif /* not USE_MAGMA_GETRI */

  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xGETRI(): LAPACK GETRF not supported on selected "
                           "location");
  }

  if (info != 0) {
    throw excMath("xGETRI(): error: info = %d", info);
  }
#endif

};
/** \overload
 *
 *  \param[in]        A
 *                    Matrix in LU decomposition.
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). The CUBLAS backend
 *                    allows specifying an empty matrix for ipiv, assuming that
 *                    the input is a non-pivoting LU decomposition (see xGETRF).
 */
template <typename T>
inline void xGETRI(Dense<T>& A, Dense<int>& ipiv) {
  Dense<T> work;
  xGETRI(A, ipiv, work);
};

/** \brief            Compute the inverse using the LU decomposition of a
 *                    matrix out-of-place
 *
 *  \note             This function is provided to suit the CUBLAS
 *                    implementation of an out-of-place inversion. An 
 *                    out-of-place inversion of matrices in main memory
 *                    or using the MAGMA getri() implementation requires extra 
 *                    memory and an additional memory copy as the standard 
 *                    LAPACK inversion function is in-place.
 *
 *  C = A**-1 (out-of-place)
 *
 *  \param[in]        A
 *                    Matrix in LU decomposition.
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). The CUBLAS backend
 *                    allows specifying an empty matrix for ipiv, assuming that
 *                    the input is a non-pivoting LU decomposition (see xGETRF).
 *
 *  \param[in]        work
 *                    For the matrices in main memory \<work\> is a vector of 
 *                    length at least A.rows(). The optimal length can be 
 *                    determined using LAPACK's ILAENV. Some backends (e.g.  
 *                    the CUBLAS backend) don't require a preallocated work in 
 *                    which case the supplied vector is ignored. If the 
 *                    backend does require a preallocated work and none or an 
 *                    empty one is specified, the routine will allocate one of 
 *                    the optimal size.
 *
 *  \param[in]        C
 *                    The target of the out-of-place inversion. If an empty 
 *                    matrix is given as argument, the routine allocates a 
 *                    suitable one.
 *
 *  \note             The return value of the routine is checked and a
 *                    corresponding exception is thrown if the matrix is
 *                    singular.
 *
 *  \todo             The fastest way to swap two matrices on the CPU would
 *                    probably be to swap them in reasonably sized blocks (8 
 *                    or 16 elements vectorized) using a swap without 
 *                    temporary (uses no extra memory and only reads/writes 
 *                    each element once from RAM and twice in registers).
 */
template <typename T>
inline void xGETRI_oop(Dense<T>& A, Dense<int>& ipiv, Dense<T>& work,
                       Dense<T>& C) {

  if (C._rows == 0) {
    C.reallocate(A._rows, A._cols, A._location, A._device_id);
  }

#ifndef LINALG_NO_CHECKS
  // Check A
  check_input_transposed(A, "xGETRI(A, ipiv, work), A");
  if (A._rows != A._cols) {
    throw excBadArgument("xGETRI(A, ipiv, work), A: matrix must be square");
  }

  // Check if ipiv is empty _and_ we use the CUBLAS backend
  bool ipiv_empty;
#ifndef HAVE_CUDA
  ipiv_empty = false;
#else
  ipiv_empty = (ipiv._rows == 0) ? true : false;
#ifndef USE_MAGMA_GETRI
  if (ipiv_empty && A._location != Location::GPU) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: empty ipiv is only "
                         "allowed when inverting matrices on the GPU");
  }
#else
  if (ipiv_empty && A._location == Location::GPU) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: empty ipiv is not "
                         "supported when using the MAGMA backend");
  }
  if (ipiv._location != Location::host) {
    throw excBadArgument("xGETRI(A, ipiv, work), ipiv: ipiv must be allocated "
                         "in main memory for MAGMA backend");
  }
#endif /* USE_MAGMA_GETRI */
#endif /* HAVE_CUDA */

  if (!ipiv_empty) {

#ifndef USE_MAGMA_GETRI
    check_device(A, ipiv, "xGETRI()");
#endif
    check_input_transposed(ipiv, "xGETRI(A, ipiv, work), ipiv");
    if (A.rows() != ipiv.rows()) {
      throw excBadArgument("xGETRI(A, ipiv, work), ipiv: argument matrix size "
                           "mismatch: if ipiv is not empty, A.rows() = %d must "
                           "equal ipiv.rows() = %d", A.rows(), ipiv.rows());
    }
  
  }

  // Check C
  check_device(A, C, "xGETRI()");
  check_output_transposed(C, "xGETRI()");
  if (A._rows != C._rows || A._cols != C._cols) {
    throw excBadArgument("xGETRI(A, ipiv, work, C), C: dimensions of A (%dx%d) "
                         "and C (%dx%d) don't match", A.rows(), A.cols(),
                         C.rows(), C.cols());
  }
#endif /* not LINALG_NO_CHECKS */

  auto device_id = A._device_id;
  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto ipiv_ptr = ipiv._begin();

  if (A._location == Location::host) {

    auto lwork = work._rows;

    // If work is empty, we have to allocate it optimally
    if (lwork == 0) {

      using LinAlg::Type;
      using LinAlg::LAPACK::FORTRAN::ILAENV;

      switch (type<T>()) {
        case Type::S:
          lwork = ILAENV(1, "sgetri", "", n, -1, -1, -1);
          break;
        case Type::D:
          lwork = ILAENV(1, "dgetri", "", n, -1, -1, -1);
          break;
        case Type::C:
          lwork = ILAENV(1, "cgetri", "", n, -1, -1, -1);
          break;
        case Type::Z:
          lwork = ILAENV(1, "zgetri", "", n, -1, -1, -1);
          break;
        default:
          throw excBadArgument("xGETRI(): unsupported data type");
      }

      work.reallocate(lwork, 1, Location::host, 0);
      lwork = work._rows;

    }

    // At least the temporary could be omitted if we had a fast swap for 
    // arrays
    printf("LinAlg::LAPACK::xGETRI(): warning, out-of-place inversion for "
           "matrices in main memory currently requires 3 times the memory of "
           "an in-place inversion and incurs three memory copies of the "
           "matrix\n");
    Dense<T> A_tmp;
    A_tmp << A;

    auto work_ptr = work._begin();
    int  info     = 0;

    FORTRAN::xGETRI(n, A_ptr, lda, ipiv_ptr, work_ptr, lwork, &info);

#ifndef LINALG_NO_CHECKS
    if (info != 0) {
      throw excMath("xGETRI(): error: info = %d", info);
    }
#endif

    C << A;
    A << A_tmp;

  }
#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

#ifndef USE_MAGMA_GETRI
    auto ipiv_ptr = (ipiv_empty) ? nullptr : ipiv._begin();
    auto C_ptr    = C._begin();
    auto ldc      = C._leading_dimension;
    int  info     = 0;

    CUBLAS::xGETRI(LinAlg::CUDA::CUBLAS::handles[device_id], n, A_ptr, lda,
                   ipiv_ptr, C_ptr, ldc, &info);

#else /* USE_MAGMA_GETRI */

    auto lwork = work._rows;

    // If work is empty, we have to allocate it optimally (in main memory!)
    if (lwork == 0) {

      using LinAlg::Type;
      using LinAlg::LAPACK::FORTRAN::ILAENV;

      lwork = MAGMA::get_xgetri_nb<T>(n) * n;

      work.reallocate(lwork, 1);

    } 

    // At least the temporary could be omitted if we had a fast swap for 
    // arrays
    printf("LinAlg::LAPACK::xGETRI(): warning, using xGETRI() with the MAGMA "
           "backend: out-of-place inversion for matrices in requires 3 times "
           "the memory of an in-place inversion and incurs three memory copies "
           "of the matrix\n");
    Dense<T> A_tmp;
    A_tmp << A;

    auto work_ptr = work._begin();
    int  info     = 0;

    MAGMA::xGETRI(n, A_ptr, lda, ipiv_ptr, work_ptr, lwork, &info);

#ifndef LINALG_NO_CHECKS
    if (info != 0) {
      throw excMath("xGETRI(): error: info = %d", info);
    }
#endif

    C << A;
    A << A_tmp;

#endif /* not USE_MAGMA_GETRI */

#ifndef LINALG_NO_CHECKS
    if (info != 0) {
      throw excMath("xGETRI(): error: info = %d", info);
    }
#endif

  }
#endif

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xGETRI(): LAPACK GETRF not supported on selected "
                           "location");
  }
#endif

};

/** \overload
 *
 *  \param[in]        A
 *                    Matrix in non-pivoted LU decomposition.
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). The CUBLAS backend
 *                    allows specifying an empty matrix for ipiv, assuming that
 *                    the input is a non-pivoting LU decomposition (see xGETRF).
 *
 *  \param[in]        C
 *                    The target of the out-of-place inversion. If an empty 
 *                    matrix is given as argument, the routine allocates a 
 *                    suitable one.
 */
template <typename T>
inline void xGETRI_oop(Dense<T>& A, Dense<int>& ipiv, Dense<T>& C) {
  Dense<T>   work;
  xGETRI(A, ipiv, work, C);
};

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_GETRI_H_ */
