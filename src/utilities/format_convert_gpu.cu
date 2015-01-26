/** \file
 *
 *  \brief            Routines for converting formats on GPUs
 *
 *  \date             Created:  Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "preprocessor.h"

namespace LinAlg {

namespace Utilities {

// Declaration of kernels for converting complex to pairs of real matrices and 
// vice versa
__global__ void gpu_complex2realimag_css_kernel(int, int, float2*, int, float*, 
                                               int, float*, int);
__global__ void gpu_complex2realimag_zdd_kernel(int, int, double2*, int, 
                                               double*, int, double*, int);
__global__ void gpu_realimag2complex_ssc_kernel(int, int, float, float*, int, 
                                               float*, int, float2, float2*, 
                                               int);
__global__ void gpu_realimag2complex_ddz_kernel(int, int, double, double*, int,
                                               double*, int, double2, double2*, 
                                               int);

/** \brief            Routine to convert a complex matrix to a pair of dense 
 *                    matrices
 *
 *  C -> A + i * B
 *
 *  \param[in]        line_length
 *                    rows for ColMajor, columns for RowMajor
 *
 *  \param[in]        lines
 *                    columns for ColMajor, rows for RowMajor
 *
 *  \param[in]        beta
 *
 *  \param[in]        C_dev
 *
 *  \param[in]        ldc
 *
 *  \param[in,out]    A_dev
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    B_dev
 *
 *  \param[in]        ldb
 */
void gpu_complex2realimag(int line_length, int lines, float2* C_dev, int ldc, 
                          float* A_dev, int lda, float* B_dev, int ldb) {

  unsigned int n_threads = lines + (CUDA_BLOCK_SIZE -
                                                (lines % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_complex2realimag_css_kernel<<< n_blocks, CUDA_BLOCK_SIZE >>>(
          line_length, lines, C_dev, ldc, A_dev, lda, B_dev, ldb);

}
/** \overload
 */
void gpu_complex2realimag(int line_length, int lines, double2* C_dev, int ldc, 
                          double* A_dev, int lda, double* B_dev, int ldb) {

  unsigned int n_threads = lines + (CUDA_BLOCK_SIZE -
                                                (lines % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_complex2realimag_zdd_kernel<<< n_blocks, CUDA_BLOCK_SIZE >>>(
          line_length, lines, C_dev, ldc, A_dev, lda, B_dev, ldb);

}
#ifndef DOXYGEN_SKIP
__global__ void gpu_complex2realimag_css_kernel(int line_length, int lines,
                                                float2* C_dev, int ldc,
                                                float*  A_dev, int lda,
                                                float*  B_dev, int ldb) {

  int line = blockIdx.x * blockDim.x + threadIdx.x;

  if (line < lines) {

    for (int element = 0; element < line_length; ++element) {

      A_dev[line * lda + element] = C_dev[line * ldc + element].x;
      B_dev[line * ldb + element] = C_dev[line * ldc + element].y;
    
    }
  
  }

  __syncthreads();

}
__global__ void gpu_complex2realimag_zdd_kernel(int line_length, int lines,
                                                double2* C_dev, int ldc,
                                                double*  A_dev, int lda,
                                                double*  B_dev, int ldb) {

  int line = blockIdx.x * blockDim.x + threadIdx.x;

  if (line < lines) {

    for (int element = 0; element < line_length; ++element) {

      A_dev[line * lda + element] = C_dev[line * ldc + element].x;
      B_dev[line * ldb + element] = C_dev[line * ldc + element].y;
    
    }
  
  }

  __syncthreads();

}
#endif

/** \brief            Routine to convert a pair of real matrices to a complex 
 *                    matrix
 *
 *  C <- alpha * (A + i * B) + beta * C
 *
 *  \param[in]        line_length
 *                    rows for ColMajor, columns for RowMajor
 *
 *  \param[in]        lines
 *                    columns for ColMajor, rows for RowMajor
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A_dev
 *
 *  \param[in]        lda
 *
 *  \param[in]        B_dev
 *
 *  \param[in]        ldb
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C_dev
 *
 *  \param[in]        ldc
 */
void gpu_realimag2complex(int line_length, int lines, float alpha, float* 
                          A_dev, int lda, float* B_dev, int ldb, float2 beta, 
                          float2* C_dev, int ldc) {

  unsigned int n_threads = lines + (CUDA_BLOCK_SIZE -
                                                (lines % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_realimag2complex_ssc_kernel<<< n_blocks, CUDA_BLOCK_SIZE >>>(
          line_length, lines, alpha, A_dev, lda, B_dev, ldb, beta, C_dev, ldc);

}
/** \overload
 */
void gpu_realimag2complex(int line_length, int lines, double alpha, double* 
                          A_dev, int lda, double* B_dev, int ldb,
                          double2 beta, double2* C_dev, int ldc) {

  unsigned int n_threads = lines + (CUDA_BLOCK_SIZE -
                                                (lines % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_realimag2complex_ddz_kernel<<< n_blocks, CUDA_BLOCK_SIZE >>>(
          line_length, lines, alpha, A_dev, lda, B_dev, ldb, beta, C_dev, ldc);

}
#ifndef DOXYGEN_SKIP
__global__ void gpu_realimag2complex_ssc_kernel(int line_length, int lines,
                                                float alpha, float*  A_dev,
                                                int lda, float*  B_dev, int ldb,
                                                float2 beta, float2* C_dev,
                                                int ldc) {

  int line = blockIdx.x * blockDim.x + threadIdx.x;

  float A, B;
  float2 C, betaC;

  if (line < lines) {

    for (int element = 0; element < line_length; ++element) {

      A     = A_dev[line * lda + element];
      B     = B_dev[line * ldb + element];
      C     = C_dev[line * ldc + element];
      betaC = cuCmulf(beta, C);

      C_dev[line * ldc + element].x = alpha * A + betaC.x;
      C_dev[line * ldc + element].y = alpha * B + betaC.y;
    
    }
  
  }

  __syncthreads();

}
__global__ void gpu_realimag2complex_ddz_kernel(int line_length, int lines,
                                                double alpha, double*  A_dev,
                                                int lda, double*  B_dev, int ldb,
                                                double2 beta, double2* C_dev,
                                                int ldc) {

  int line = blockIdx.x * blockDim.x + threadIdx.x;

  double A, B;
  double2 C, betaC;

  if (line < lines) {

    for (int element = 0; element < line_length; ++element) {

      A = A_dev[line * lda + element];
      B = B_dev[line * ldb + element];
      C = C_dev[line * ldc + element];
      betaC = cuCmul(beta, C);

      C_dev[line * ldc + element].x = alpha * A + betaC.x;
      C_dev[line * ldc + element].y = alpha * B + betaC.y;
    
    }
  
  }

  __syncthreads();

}
#endif



// Declaration of kernels
__global__ void gpu_csr2dense_d_kernel(double*, int*, int*, int, int, double*,
                                       int);

/** \brief            Routine to convert a CSR matrix to a dense matrix in 
 *                    Format::ColMajor
 *
 *  \param[in]        values_dev
 *
 *  \param[in]        indices_dev
 *
 *  \param[in]        edges_dev
 *
 *  \param[in]        rows
 *
 *  \param[in]        first_index
 *
 *  \param[out]       dense_dev
 *
 *  \param[in]        leading_dimension
 */
void gpu_csr2dense(double* values_dev, int* indices_dev,
                   int* edges_dev, int rows, int first_index,
                   double* dense_dev, int leading_dimension) {

  unsigned int n_threads = rows + (CUDA_BLOCK_SIZE - (rows % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_csr2dense_d_kernel<<< n_blocks, CUDA_BLOCK_SIZE >>>(values_dev,
          indices_dev, edges_dev, rows, first_index, dense_dev, 
          leading_dimension);

}

/** \brief            Routine to asynchronously convert a CSR matrix to a dense 
 *                    matrix in Format::ColMajor
 *
 *  \param[in]        values_dev
 *
 *  \param[in]        indices_dev
 *
 *  \param[in]        edges_dev
 *
 *  \param[in]        rows
 *
 *  \param[in]        first_index
 *
 *  \param[out]       dense_dev
 *
 *  \param[in]        leading_dimension
 *
 *  \param[in]        cuda_stream
 */
void gpu_csr2dense_async(double* values_dev, int* indices_dev,
                         int* edges_dev, int rows, int first_index,
                         double* dense_dev, int leading_dimension,
                         cudaStream_t cuda_stream) {

  unsigned int n_threads = rows + (CUDA_BLOCK_SIZE - (rows % CUDA_BLOCK_SIZE));
           int n_blocks  = n_threads / CUDA_BLOCK_SIZE;

  gpu_csr2dense_d_kernel<<< n_blocks, CUDA_BLOCK_SIZE, 0, cuda_stream >>>(
          values_dev, indices_dev, edges_dev, rows, first_index, dense_dev, 
          leading_dimension);

}
#ifndef DOXYGEN_SKIP
__global__ void gpu_csr2dense_d_kernel(double* values_dev,
                                       int* indices_dev,
                                       int* edges_dev, int rows,
                                       int first_index,
                                       double* dense_dev,
                                       int leading_dimension) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {

    int row_start = edges_dev[row] - first_index;
    int row_stop  = edges_dev[row + 1] - first_index;
  
    for (int index = row_start; index < row_stop; ++index) {

      int col = indices_dev[index] - first_index;
    
      dense_dev[leading_dimension * col + row] = values_dev[index];
    
    }
  
  }

  __syncthreads();

}

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */
#endif
