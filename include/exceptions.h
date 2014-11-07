/** \file
 *
 *  \brief            Exceptions
 *
 *  \date             Created: Jul 12, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

#ifndef LINALG_EXCEPTIONS_H_
#define LINALG_EXCEPTIONS_H_

#ifndef LINALG_NO_CHECKS

//#define EXCEPTION_STACK_TRACE
#ifdef EXCEPTION_STACK_TRACE
#include <execinfo.h> // backtrace, backtrace_symbols
#include <cxxabi.h>   // __cxa_demangle
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <exception>
#include <string>
#include <sstream>    // std::stringstream

#include "utilities/stringformat.h"

/// Maximal depth of the stack for the stack trace
#define MAX_STACK 50

namespace LinAlg {

/** \exception        excBaseException
 *
 *  \brief            Virtual exception base class.
 *
 *  Notable features are construction from format string and display of a
 *  stack trace from the throwing point for easier debugging.
 */
class excLinAlg : public std::exception {

 public:

#ifndef DOXYGEN_SKIP
  virtual const char* prefix() const = 0;
  virtual const char* suffix() const { return ""; }

  std::string usermsg;
  std::string stack_trace;
  void get_stack();

  // Concatenate the exception's prefix and the user supplied string and an
  // optionally overridden suffix()
  const char* what() const throw() {
    std::string error_msg = std::string(prefix()) + usermsg +
                            std::string(suffix()) + stack_trace;
    return error_msg.c_str();
  }
#endif /* DOXYGEN_SKIP */

  /** \brief          Constructor from const char*
   *
   *  The resulting string will be returned from what().
   *
   *  \param[in]      userformatstring
   *                  printf-style format string
   */
  excLinAlg(const char* userstring) {
    usermsg = std::string(userstring);
    get_stack();
  }

  /** \brief          Constructor from formatstring
   *
   *  The resulting string will be returned from what().
   *
   *  \param[in]      userformatstring
   *                  printf-style format string
   *
   *  \param[in]      userformatargs
   *                  printf-style format arguments
   */
  template <typename... Ts>
  excLinAlg(const char* userformatstring, Ts... userformatargs) {
    usermsg = Utilities::stringformat(userformatstring, userformatargs...);
    get_stack();
  }

  /** \brief          Constructor from std::string
   *
   *  The resulting string will be returned from what().
   *
   *  \param[in]      string
   *                  Suffix that is appended to a string describing the type
   *                  of exception.
   */
  excLinAlg(std::string string) {
    usermsg = string;
    get_stack();
  }

  ~excLinAlg() throw() {}

};

inline void excLinAlg::get_stack() {

#ifndef EXCEPTION_STACK_TRACE
  // A function that does nothing
#else
  // A function that gets a stack trace at the time of the construction of the 
  // exception.

  // Create a buffer and fill it with pointers to stack frames
  void* stack_pointers[MAX_STACK];
  auto stack_size = backtrace(stack_pointers, MAX_STACK);

  // Convert to C strings (requires debug symbols (-g when compiling) and
  // will result in mangled names due to C++ name mangling)
  char** mangled_function_names;
  mangled_function_names = backtrace_symbols(stack_pointers, stack_size);

  // Unmangle the C++ names
  std::vector<char*> function_names(stack_size);
  std::vector<int> statuses;
  for (int i = 0; i < stack_size; ++i) {

    function_names[i] = abi::__cxa_demangle(mangled_function_names[i], 0, 0,
                                            &statuses[i]);

  }
  free(mangled_function_names);

  // Construct the final string
  std::stringstream stack_trace_stream;
  stack_trace_stream << "\nStack when exception was thrown:\n";

  for (int i = 0; i < stack_size; ++i) {

    if (statuses[i] == 0) {

      stack_trace_stream << "\t" << function_names[i] << "\n";

      free(function_names[i]);

    } else {

      stack_trace_stream << "\t" << mangled_function_names[i]
                         << "  (unable to demangle)\n";

    }

  }

  stack_trace = stack_trace_stream.str();
#endif /* not EXCEPTION_STACK_TRACE */

}


/** \exception        excUnimplemented
 *
 *  \brief            Exception to throw to signal unimplemented
 *                    functionality.
 */
class excUnimplemented : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excUnimplemented(const char* userfmtstr, Ts... args)
                 : excLinAlg(userfmtstr, args...) {}
  excUnimplemented(const char* userstring) : excLinAlg(userstring) {}
  excUnimplemented(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg unimplemented - "; }
#endif
};

/** \exception        excBadArgument
 *
 *  \brief            Exception to signal that there's an error in the
 *                    arguments provided.
 */
class excBadArgument : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excBadArgument(const char* userfmtstr, Ts... args)
               : excLinAlg(userfmtstr, args...) {}
  excBadArgument(const char* userstring) : excLinAlg(userstring) {}
  excBadArgument(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg bad argument - "; }
#endif
};

/** \exception        excMallocError
 *
 *  \brief            Exception to signal that there was an error while
 *                    allocating memory.
 */
class excMallocError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excMallocError(const char* userfmtstr, Ts... args)
               : excLinAlg(userfmtstr, args...) {}
  excMallocError(const char* userstring) : excLinAlg(userstring) {}
  excMallocError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg malloc error - "; }
#endif
};

/** \exception        excSystemError
 *
 *  \brief            Exception to signal that there was a system error.
 */
class excSystemError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excSystemError(const char* userfmtstr, Ts... args)
               : excLinAlg(userfmtstr, args...) {}
  excSystemError(const char* userstring) : excLinAlg(userstring) {}
  excSystemError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg system error - "; }
#endif
};

/** \exception        excBadFile
 *
 *  \brief            Exception to signal that a file has invalid syntax.
 */
class excBadFile : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excBadFile(const char* userfmtstr, Ts... args)
           : excLinAlg(userfmtstr, args...) {}
  excBadFile(const char* userstring) : excLinAlg(userstring) {}
  excBadFile(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg bad file - "; }
#endif
};

/** \exception        excBufferHelper
 *
 *  \brief            Exception to signal that a problem in the BufferHelper 
 *                    occurred
 */
class excBufferHelper : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excBufferHelper(const char* userfmtstr, Ts... args)
           : excLinAlg(userfmtstr, args...) {}
  excBufferHelper(const char* userstring) : excLinAlg(userstring) {}
  excBufferHelper(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg buffer - "; }
#endif
};

/** \exception        excMath
 *
 *  \brief            Exception to signal a mathematical problem.
 */
class excMath : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excMath(const char* userfmtstr, Ts... args)
           : excLinAlg(userfmtstr, args...) {}
  excMath(const char* userstring) : excLinAlg(userstring) {}
  excMath(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg math - "; }
#endif
};

#ifdef HAVE_CUDA
namespace CUDA {

/** \exception        excCUDAError
 *
 *  \brief            Exception to signal that there was an error while
 *                    executing a CUDA routine.
 */
class excCUDAError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excCUDAError(const char* userfmtstr, Ts... args)
             : excLinAlg(userfmtstr, args...) {}
  excCUDAError(const char* userstring) : excLinAlg(userstring) {}
  excCUDAError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg CUDA error - "; }
#endif
};

/** \exception        excCUBLASError
 *
 *  \brief            Exception to signal that there was an error while
 *                    executing a CUBLAS routine.
 */
class excCUBLASError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excCUBLASError(const char* userfmtstr, Ts... args)
               : excLinAlg(userfmtstr, args...) {}
  excCUBLASError(const char* userstring) : excLinAlg(userstring) {}
  excCUBLASError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg CUBLAS error - "; }
#endif
};

/** \exception        excCUSPARSEError
 *
 *  \brief            Exception to signal that there was an error while
 *                    executing a CUSPARSE routine..
 */
class excCUSPARSEError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excCUSPARSEError(const char* userfmtstr, Ts... args)
                 : excLinAlg(userfmtstr, args...) {}
  excCUSPARSEError(const char* userstring) : excLinAlg(userstring) {}
  excCUSPARSEError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg CUSPARSE error - "; }
#endif
};

} /* namespace LinAlg::CUDA */
#endif

#ifdef HAVE_MPI
namespace MPI {

/** \exception        excMPIError
 *
 *  \brief            Exception to signal that there was an error in an MPI
 *                    call.
 */
class excMPIError : public excLinAlg {
#ifndef DOXYGEN_SKIP
 public:
  template <typename... Ts>
  excMPIError(const char* userfmtstr, Ts... args)
             : excLinAlg(userfmtstr, args...) {}
  excMPIError(const char* userstring) : excLinAlg(userstring) {}
  excMPIError(std::string string) : excLinAlg(string) {}
  const char* prefix() const { return "LinAlg MPI error - "; }
  const char* suffix() const { return suffix_string.c_str(); }

  std::string suffix_string;
#endif

  /** \brief          Routine to extract all useful information from an
   *                  MPI_Status and set the suffix of the user message
   *                  accordingly.
   *
   *  \param[in]      mpi_status
   *                  The status struct to extract the information from.
   */
  void set_status(MPI_Status mpi_status) {

    auto source_rank = mpi_status.MPI_SOURCE;
    auto tag = mpi_status.MPI_TAG;
    auto mpi_error = mpi_status.MPI_ERROR;

    // Get the MPI error string, construct stringstream with apropriate content
    char* error_string = new char[MPI_MAX_ERROR_STRING];
    int error_string_length;
    MPI_Error_string(mpi_error, error_string, &error_string_length);

    std::stringstream suffix_sstream;
    suffix_sstream << "(source rank: " << source_rank
                   << ", tag: " << tag
                   << ", error string: " << error_string << ")";

    delete[] error_string;

    suffix_string = suffix_sstream.str();

  }

};

} /* namespace LinAlg::MPI */
#endif

} /* namespace LinAlg */

#endif /* LINALG_NO_CHECKS */

#endif /* LINALG_EXCEPTIONS_H_ */
