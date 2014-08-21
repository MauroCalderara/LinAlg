/** \file             getri.h
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
#include "cuda_helper.h"

#ifdef HAVE_MAGMA
#include <magma.h>
#include <magma_lapack.h>
#endif

#endif


#include "types.h"
#include "exceptions.h"
#include "utilities/checks.h"
#include "dense.h"

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

#ifndef DOXYGEN_SKIP
extern C {
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

/** \brief            GETRI
 *
 *  P * L * U <- A^{-1}
 *
 *  \param[in]        n
 *
 *  \param[in|out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in|out]    ipiv
 *
 *  \param[in]        work
 *
 *  \param[in]        lwork
 *
 *  \param[in|out]    info
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
 *  \param[in|out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in|out]    ipiv
 *
 *  \param[in|out]    C
 *
 *  \param[in]        ldc
 *
 *  \param[in|out]    info
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
 *  \param[in|out]    Aarray
 *
 *  \param[in]        lda
 *
 *  \param[in|out]    PivotArray
 *
 *  \param[in|out]    Carray
 *
 *  \param[in]        ldc
 *
 *  \param[in|out]    infoArray
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
#endif

#ifdef HAVE_MAGMA
namespace MAGMA {

} /* namespace LinAlg::LAPACK::MAGMA */
#endif /* HAVE_MAGMA */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_input_transposed;

/** \brief            Compute the inverse using the LU decomposition of a
 *                    matrix in-place or out-of-place
 *
 *  Note that in-place inversion on GPU using CUBLAS matrices requires extra
 *  memory and an additional memory copy as the CUBLAS inversion function is
 *  out-of-place. An out-of-place inversion of matrices in main memory requires
 *  extra memory and an additional memory copy as the LAPACK inversion function
 *  is in-place.
 *
 *  A = A**-1 (in-place)
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
 *                    OPTIONAL: For the matrices in main memory work is a vector
 *                    of length at least A.rows(). The optimal length can be
 *                    determined using LAPACK's ILAENV. Some backends (e.g. the
 *                    CUBLAS backend) don't require a preallocated work in which
 *                    case the argument is ignored. If the backend does require
 *                    a preallocated work and none is specified, the routine
 *                    will allocate one of the optimal size.
 *
 *  \param[in]        C
 *                    OPTIONAL: if unspecified or empty, an in-place inversion
 *                    is performed, if specified, an out-of-place inversion is
 *                    performed.
 *
 *  \note             The return value of the routine is checked and a
 *                    corresponding exception is thrown if the matrix is
 *                    singular.
 */
template <typename T>
inline void xGETRI(Dense<T>& A, Dense<int>& ipiv, Dense<T>& work, Dense<T>& C) {

#ifndef LINALG_NO_CHECKS
  // Check the inputs
  check_device(A, ipiv, "xGETRI()");
  check_input_transposed(A, "xGETRI()");
#ifndef HAVE_CUDA
  check_input_transposed(ipiv, "xGETRI()");
#else
  bool ipiv_empty = (ipiv._rows == 0) ? true : false;
  if (!ipiv_empty) {
    check_input_transposed(ipiv, "xGETRI()");
  }
#endif
  if (A._rows != A._cols) {
    throw excBadArgument("xGETRI(): matrix A must be a square matrix");
  }

  // Check ipiv specifically
#ifndef HAVE_CUDA
  if (A.rows() != ipiv.rows()) {
    throw excBadArgument("xGETRI(): argument matrix size mismatch: "
                         "matrix.rows() != ipiv.rows() ");
  }
#else
  if (A.rows() != A.cols() && A._location == Location::GPU) {
    throw excBadArgument("xGETRI(): matrix A must be a square matrix");
  }
  if (!ipiv_empty) {
    if (A.rows() != ipiv.rows()) {
      throw excBadArgument("xGETRI(): argument matrix size mismatch: "
                           "if ipiv is not empty, matrix.rows() must equal "
                           "ipiv.rows()");
    }
  }
#endif
#endif /* LINALG_NO_CHECKS */

  // Check C
  bool out_of_place = (C._rows == 0) ? false : true;
#ifndef LINALG_NO_CHECKS
  if (out_of_place) {
    check_device(A, C, "xGETRI()");
    check_output_transposed(C, "xGETRI()");
    if (A._rows != C._rows || A._cols != C._cols) {
      throw excBadArgument("xGETRI(): (out-of-place) dimensions of A and C "
                           "don't match");
    }
  }
#endif


  auto device_id = A._device_id;
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

      switch (type<T>()) {
        case Type::S:
          lwork = LAPACK::ILAENV(1, "sgetri", "", n, -1, -1, -1);
          break;
        case Type::D:
          lwork = LAPACK::ILAENV(1, "dgetri", "", n, -1, -1, -1);
          break;
        case Type::C:
          lwork = LAPACK::ILAENV(1, "cgetri", "", n, -1, -1, -1);
          break;
        case Type::Z:
          lwork = LAPACK::ILAENV(1, "zgetri", "", n, -1, -1, -1);
          break;
        default:
          throw excBadArgument("xGETRI(): unsupported data type");
      }

      work.reallocate(lwork, 1);
      lwork = work._rows;

    }

    // TODO: this would be faster using the swap-without-temporary trick but I
    // don't yet have matrix addition and subtraction
    Dense<T> A_tmp;

    if (out_of_place) {
      printf("LinAlg::LAPACK::xGETRI(): warning, out-of-place inversion for "
             "matrices in main memory requires 3 times the memory of in-place "
             "inversion and incurs three memory copies of the matrix\n");
      A_tmp << A;
    }

    auto work_ptr = work._begin();
    FORTRAN::xGETRI(n, A_ptr, lda, ipiv_ptr, work_ptr, lwork, &info);

    if (out_of_place) {
      C << A;
      A << A_tmp;
    }

  }
#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

    // CUBLAS' getri is out-of-place so we need to allocate a C and then stream
    // it back into A after the operation

    if (!out_of_place) {
      C.reallocate(A._rows, A._cols, A._location, A._device_id);
    }

    auto C_ptr = C._begin();
    auto ldc = C._leading_dimension;

    using LinAlg::CUDA::CUBLAS::handles;
    CUBLAS::xGETRI(handles[device_id], n, A_ptr, lda, ipiv_ptr, C_ptr, ldc,
                   &info);

    if (!out_of_place) {
      A << C;
    }

  }
#ifdef HAVE_MAGMA
  // check if MAGMA's or CUBLAS' GETRI is faster and use that one.
#endif
#endif

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
 */
template <typename T>
inline void xGETRI(Dense<T>& A, Dense<int>& ipiv, Dense<T>& work) {
  Dense<T> C;
  xGETRI(A, ipiv, work, C);
};
/** \overload
 */
template <typename T>
inline void xGETRI(Dense<T>& A, Dense<int>& ipiv) {
  Dense<T> work;
  Dense<T> C;
  xGETRI(A, ipiv, work, C);
};


} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_GETRI_H_ */
