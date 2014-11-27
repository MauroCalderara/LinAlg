/** \file
 *
 *  \brief            A memory buffer
 *
 *  \date             Created:  Aug 10, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_BUFFER_H_
#define LINALG_UTILITIES_BUFFER_H_

#include <vector>     // std::vector
#include <functional> // std::function
#include <future>     // std::future

#include "../preprocessor.h"

#include "../types.h"
#include "../exceptions.h"
#include "../streams.h"

#ifndef DOXYGEN_SKIP

# ifdef BUFFER_HELPER_DISPLAY
#   include <iostream>     // std::cout
# endif

# ifdef BUFFER_HELPER_VERBOSE
#   include <string>       // std::string
#   include <cstdio>       // std::printf
# endif

#endif /* DOXYGEN_SKIP */

namespace LinAlg {

namespace Utilities {

/** \brief            A simple class for buffer management.
 */
struct BufferHelper {

  BufferHelper() {}
  BufferHelper(I_t size, I_t lookahead, BufferType type,
               std::function<void(I_t)> loader,
               std::function<void(I_t)> deleter);
  BufferHelper(I_t size, I_t lookahead, BufferType type,
               std::function<void(I_t)> loader,
               std::function<void(I_t)> deleter, Stream& stream);
  ~BufferHelper();

  void flush();
  void wait(I_t n);
  void preload(BufferDirection direction);

#ifndef DOXYGEN_SKIP
  // The idea being that you typically don't need to access these.
  Stream*                  _stream;
  bool                     _manage_stream;
  I_t                      _size;
  I_t                      _lookahead;
  std::function<void(I_t)> _loader;
  std::function<void(I_t)> _deleter;
  BufferType               _type;
  BufferDirection          _direction;

  bool                     _initialized;
  int                      _last_requested;
  std::vector<I_t>         _buffer_tickets;
  std::vector<int>         _buffer_status;    // -1 deleting, 0 clear,
                                              // 1 in flight, 2 loaded
# ifndef BUFFER_HELPER_VERBOSE
  template <typename... Us>
  void verbose_print(const char* formatstring, Us... formatargs) { return; }
# else 
  template <typename... Us>
  void verbose_print(const char* formatstring, Us... formatargs) {
    using LinAlg::Utilities::stringformat;
    std::string line = "BUFFER: " + stringformat(formatstring, formatargs...);
    std::printf("%s", line.c_str());
  }
# endif /* not BUFFER_HELPER_VERBOSE */
#endif /* not DOXYGEN_SKIP */

};

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_BUFFER_H_ */
