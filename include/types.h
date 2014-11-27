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

#include "elementary_types.h"
#include "exceptions.h"

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
template <typename T> inline Type type()      { return Type::O; }
/** \overload
 *
 *  \returns          Type::S
 */
template <>           inline Type type<S_t>() { return Type::S; }
/** \overload
 *
 *  \returns          Type::D
 */
template <>           inline Type type<D_t>() { return Type::D; }
/** \overload
 *
 *  \returns          Type::C
 */
template <>           inline Type type<C_t>() { return Type::C; }
/** \overload
 *
 *  \returns          Type::Z
 */
template <>           inline Type type<Z_t>() { return Type::Z; }
/** \overload
 *
 *  \returns          Type::I
 */
template <>           inline Type type<I_t>() { return Type::I; }


/** \brief            Matrix properties
 *
 *  We store matrix properties as bitfield internally. The matrices' setter 
 *  members check for internal consistency.
 */
enum Property {
  general   = 0x01,
  symmetric = 0x02,
  hermitian = 0x04,
  packed    = 0x08,
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
  ColMajor,     //< Column major (Fortran layout: [a_00, a_10, a_20, ...])
  RowMajor,     //< Row major (C/C++ layout: [a_00, a_01, a_02, ...])
  CSR,          //< Compressed Sparse Row
  CSC,          //< Compressed Sparse Column
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

/** \brief            IJ, a point in a matrix (row/column pair)
 */
struct IJ {

  I_t row;      //< Row
  I_t col;      //< Column

  IJ() : row(0), col(0) {};             //< Empty constructor
  IJ(I_t i, I_t j) : row(i), col(j) {}; //< Constructor from row and column

};
inline IJ operator+(const IJ left, const IJ right) {
  return IJ(left.row + right.row, left.col + right.col);
}
inline IJ operator-(const IJ left, const IJ right) {
  return IJ(left.row - right.row, left.col - right.col);
}
inline bool operator==(const IJ left, const IJ right) {
  return ((left.row == right.row) && (left.col == right.col));
}

/** \brief            SubBlock, a matrix subblock
 */
struct SubBlock {

  // Start values are inclusive, stop values exclusive
  I_t first_row, last_row, first_col, last_col;

  SubBlock() : first_row(0), last_row(0), first_col(0), last_col(0) {}

  SubBlock(IJ start, IJ stop)
         : first_row(start.row),
           last_row(stop.row),
           first_col(start.col),
           last_col(stop.col) {
    #ifndef LINALG_NO_CHECKS
    if (last_row < first_row || last_col < first_col) {
      throw excBadArgument("SubBlock(): invalid subblock specification");
    }
    #endif
  }

  /// Matlab style block specification
  SubBlock(I_t first_row, I_t last_row, I_t first_col, I_t last_col)
         : SubBlock(IJ(first_row, first_col), IJ(last_row, last_col)) {}

  /// Upper left corner (inclusive)
  inline IJ start() const { return IJ(first_row, first_col); }

  /// Lower right corner (exclusive)
  inline IJ stop() const { return IJ(last_row, last_col); }

  /// Rows
  inline I_t rows() const { return (last_row - first_row); }

  /// Columns
  inline I_t cols() const { return (last_col - first_col); }

};
inline SubBlock transposed(SubBlock block) {
  return SubBlock(block.first_col, block.last_col,
                  block.first_row, block.last_row);
}
inline bool operator==(const SubBlock left, const SubBlock right) {
  return ((left.first_row == right.first_row) &&
          (left.first_col == right.first_col) &&
          (left.last_row  == right.last_row)  &&
          (left.last_col  == right.last_col)    );
}
inline bool operator!=(const SubBlock left, const SubBlock right) {
  return ((left.first_row != right.first_row) ||
          (left.first_col != right.first_col) ||
          (left.last_row  != right.last_row)  ||
          (left.last_col  != right.last_col)    );
}


// BLAS / LAPACK specifiers
enum class Side { left, right };
enum class UPLO { upper, lower };
enum class Diag { unit, non_unit };

} /* namespace LinAlg */

#endif /* LINALG_TYPES_H_ */
