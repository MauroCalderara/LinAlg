/** \file
 *
 *  \brief            Routines that don't fit elsewhere.
 *
 *  \date             Created:  Jul 17, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcaldrara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_MISC_H_
#define LINALG_UTILITIES_MISC_H_

#include <fstream>    // std::fstream
#include <limits>     // std::numeric_limits
#include <string>     // std::string
#include <vector>     // std::vector

#include "../profiling.h"
#include "../exceptions.h"

namespace LinAlg {

namespace Utilities {

/** \brief            A routine that jumps to a specific line in a fstream
 *
 *  \param[in,out]    stream
 *                    The fstream on which to advance to the requested line.
 *
 *  \param[in]        line
 *                    The line to advance to. The next getline call on the
 *                    stream will return the contents of line number \<line\>.
 */
inline void go_to_line(std::ifstream& stream, unsigned int line) {

  PROFILING_FUNCTION_HEADER

  stream.seekg(std::ios::beg);

  for (unsigned int i = 0; i < line - 1; ++i) {

    stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  }

}

/** \brief            A routine to read a vector from a file
 *
 *  \param[in]        filename
 *                    The name of the file to read from. Contents can be 
 *                    whitespace separated or one per line. The type is 
 *                    inferred from the type of the vector argument.
 *
 *  \param[in|out]    vector
 *                    The vector to store the file's content.
 */
template <typename T>
void read_vector(std::string filename, std::vector<T>& vector) {

  std::ifstream file(filename);

  if (file.is_open()) {
  
    T tmp;

    while (file >> tmp) {
    
      vector.push_back(tmp);
    
    }

    file.close();
  
  } else {
  
    throw excBadArgument("read_vector(): unable to open file (%s) for "
                         "reading.", filename.c_str());
  
  }

}

} /* namespace Utilities */

} /* namespace LinAlg */

#endif /* LINALG_UTILITIES_MISC_H_ */
