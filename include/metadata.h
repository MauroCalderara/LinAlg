/** \file
 *
 *  \brief            Matrix metadata
 *
 *  \date             Created:  Aut 10, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_METADATA_H_
#define LINALG_METADATA_H_

#include <vector>     // std::vector
#include <string>     // std::string

#include "preprocessor.h"
#include "types.h"
#include "dense.h"

/** \def              META_TYPE_LENGTH
 *
 *  \brief            How many elements each type can have in it's meta data.
 */
#define META_TYPE_LENGTH 8


namespace LinAlg {

/** \brief            Matrix meta data structure
 *
 *  A struct that contains matrix meta data (format, data_type, etc.).
 */
struct MetaData {

#ifndef DOXYGEN_SKIP
  std::vector<I_t> container;
  I_t N;
#endif
  // N = META_TYPE_LENGTH
  // container[0]:      format (0=Format::ColMajor, 1=Format::RowMajor, ...)
  // container[1]:      data type (0=Types::unknown, 1=Types::C, ...)
  // container[2]:      transposed (0=false, true otherwise)
  //
  // container[N]:      rows
  // container[N+1]:    cols
  //
  // container[2N]:     size
  // container[2N+1]:   n_nonzeros
  // container[2N+2]:   first_index
  // container[2N+3]:   row_offset


  // Constructors
  MetaData() : N(META_TYPE_LENGTH) { container.resize(3 * N); }
  template <typename T> MetaData(Dense<T>& matrix);

  // Helper routines for the first 3 fields
  inline I_t format2int(Format format);
  inline Format int2format(I_t i);

  template <typename T> inline I_t type2int();
  inline Type int2type(I_t i);

  template <typename T> inline std::string type2str();
  inline std::string int2str(I_t i);

  inline I_t trans2int(bool trans);
  inline bool int2trans(I_t i);

  // Main routines
  inline void set_format(Format format);
  template <typename T> inline void set_type();
  inline void set_transposed(bool trans);
  inline void set_rows(I_t rows);
  inline void set_cols(I_t cols);

  inline I_t* data();
  inline I_t size();

  template <typename T> void apply(Dense<T>& matrix);
  template <typename T> void extract(const Dense<T>& matrix);

};

/** \brief            Constructor from matrix
 */
template <typename T>
MetaData::MetaData(Dense<T>& matrix) : MetaData() {
  extract(matrix);
}

/** \brief            Convert Format to I_t
 *
 *  \param[in]        format
 *                    Format to convert.
 *
 *  \returns          Corresponding integer.
 */
inline I_t MetaData::format2int(Format format) {

  switch (format) {
    case Format::ColMajor:
      return 0;
    case Format::RowMajor:
      return 1;
    case Format::CSR:
      return 2;
    case Format::CSC:
      return 3;
  }

  return 4;

}

/** \brief            Convert I_t to Format
 *
 *  \param[in]        i
 *                    Integer to convert.
 *
 *  \returns          Corresponding format.
 */
inline Format MetaData::int2format(I_t i) {

  switch (i) {
    case 0:
      return Format::ColMajor;
    case 1:
      return Format::RowMajor;
    case 2:
      return Format::CSR;
    case 3:
      return Format::CSC;
    default:
#ifndef LINALG_NO_CHECKS
      throw excBadArgument("MetaData.int2format(): invalid input");
#else
      return Format::ColMajor;
#endif
  }

}

/** \brief            Convert Type to I_t
 *
 *  \returns          Corresponding integer.
 */
template <typename T> inline I_t MetaData::type2int()      { return 0; }
/** \overload
 *
 *  \returns          Corresponding integer.
 */
template <>           inline I_t MetaData::type2int<S_t>() { return 1; }
/** \overload
 *
 *  \returns          Corresponding integer.
 */
template <>           inline I_t MetaData::type2int<D_t>() { return 2; }
/** \overload
 *
 *  \returns          Corresponding integer.
 */
template <>           inline I_t MetaData::type2int<C_t>() { return 3; }
/** \overload
 *
 *  \returns          Corresponding integer.
 */
template <>           inline I_t MetaData::type2int<Z_t>() { return 4; }
/** \overload
 *
 *  \returns          Corresponding integer.
 */
template <>           inline I_t MetaData::type2int<I_t>() { return 5; }


/** \brief            Convert I_t to Type
 *
 *  \param[in]        i
 *                    Integer to convert
 *
 *  \returns          Corresponding Type
 */
inline Type MetaData::int2type(I_t i) {

  switch (i) {
    case 1:
      return Type::S;
    case 2:
      return Type::D;
    case 3:
      return Type::C;
    case 4:
      return Type::Z;
    case 5:
      return Type::I;
    default:
      return Type::O;
  }

}


/** \brief            Convert Type to string
 *
 *  \returns          Corresponding integer.
 */
template <typename T> inline std::string MetaData::type2str() { 
  return std::string("other");
}
/** \overload
 *
 *  \returns          Corresponding string.
 */
template <>           inline std::string MetaData::type2str<S_t>() {
  return std::string("S_t");
}
/** \overload
 *
 *  \returns          Corresponding string.
 */
template <>           inline std::string MetaData::type2str<D_t>() {
  return std::string("D_t");
}
/** \overload
 *
 *  \returns          Corresponding string.
 */
template <>           inline std::string MetaData::type2str<C_t>() {
  return std::string("C_t");
}
/** \overload
 *
 *  \returns          Corresponding string.
 */
template <>           inline std::string MetaData::type2str<Z_t>() {
  return std::string("Z_t");
}
/** \overload
 *
 *  \returns          Corresponding string.
 */
template <>           inline std::string MetaData::type2str<I_t>() {
  return std::string("I_t");
}


/** \brief            Convert I_t to std::string
 *
 *  \param[in]        i
 *                    Integer to convert
 *
 *  \returns          Corresponding string
 */
inline std::string MetaData::int2str(I_t i) {

  switch (i) {
    case 1:
      return std::string("S_t");
    case 2:
      return std::string("D_t");
    case 3:
      return std::string("C_t");
    case 4:
      return std::string("Z_t");
    case 5:
      return std::string("I_t");
    default:
      return std::string("other");
  }

}


/** \brief            Convert transposition status to I_t
 *
 *  \param[in]        trans
 *                    Transposition status
 *
 *  \returns          Corresponding I_t
 */
inline I_t MetaData::trans2int(bool trans) {

  return (trans) ? 1 : 0;

}

/** \brief            Convert I_t to transposition status
 *
 *  \param[in]        i
 *                    Integer to convert
 *
 *  \returns          Corresponding transposition status
 */
inline bool MetaData::int2trans(I_t i) {

  return (i) ? true : false;

}

/** \brief            Set format field
 *
 *  \param[in]        format
 *                    Value to set field to
 */
inline void MetaData::set_format(Format format) {

  container[0] = format2int(format);

}

/** \brief            Set type field to the type of the instance of the
 *                    template
 */
template <typename T>
inline void MetaData::set_type() {

  container[1] = type2int<T>();

}

/** \brief            Set transposition field
 *
 *  \param[in]        trans
 *                    Value to set field to
 */
inline void MetaData::set_transposed(bool trans) {

  container[2] = trans2int(trans);

}

/** \brief            Set row field
 *
 *  \param[in]        rows
 *                    Value to set field to
 */
inline void MetaData::set_rows(I_t rows) {

  container[N] = rows;

}

/** \brief            Set column field
 *
 *  \param[in]        cols
 *                    Value to set field to
 */
inline void MetaData::set_cols(I_t cols) {

  container[N + 1] = cols;

}

/** \brief            Return a pointer to the array that contains the meta
 *                    data
 */
inline I_t* MetaData::data() {

  return container.data();

}

/** \brief            Return the size of the meta data array
 */
inline I_t MetaData::size() {

  return container.size();

}


/** \brief            Extract meta data from a matrix
 *
 *  \param[in]        matrix
 *                    Matrix to read the meta data from
 */
template <typename T>
void MetaData::extract(const Dense<T>& matrix) {

  container[0] = format2int(matrix._format);
  container[1] = type2int<T>();
  container[2] = trans2int(matrix._transposed);

  container[N] = matrix._rows;
  container[N + 1] = matrix._cols;

}

/** \brief            Apply meta data to a matrix
 *
 *  \param[in]        matrix
 *                    Matrix to apply the meta data to
 */
template <typename T>
void MetaData::apply(Dense<T>& matrix) {

  I_t matrix_type = type2int<T>();

#ifndef LINALG_NO_CHECKS
  if (container[1] != matrix_type) {

    // TODO: doesn't display 4th argument properly
    throw excBadArgument("MetaData.apply(): data type stored in meta data "
                         "(%d -> %s) doesn't agree with type of matrix to "
                         "which it is to be applied (%d -> %s)", container[1], 
                         int2str(container[1]).c_str(), matrix_type,
                         type2str<T>().c_str());

  } else if (container[0] != 0 && container[0] != 1) {

    throw excBadArgument("MetaData.apply(): meta data is for a dense matrix, "
                         "can only apply to a Dense<T> matrix");

  }
#endif

  matrix._format = (container[0] == 0) ? Format::ColMajor : Format::RowMajor;
  matrix._transposed = int2trans(container[2]);

  auto rows = container[N];
  auto cols = container[N + 1];
  matrix.reallocate(rows, cols, matrix._location, matrix._device_id);

}

} /* namespace LinAlg */

#endif /* LINALG_METADATA_H_ */
