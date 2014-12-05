/** \file             profiling.h
 *
 *  \brief            Macros for profiling
 *
 *  Long description of file
 *
 *  \date             Created:  Nov  6, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          camauro <camauro@domain.tld>
 *
 *  \version          $Revision$
 */
#ifndef PROFILING_H_
#define PROFILING_H_

#include "preprocessor.h"

#ifdef SIMPLE_PROFILER
# include <atomic>
# include <unordered_map>
# include <vector>
# include <chrono>
# include <utility>
# include "threads.h"
#endif


// In terms of external profiling libraries, currently only score-p based 
// profiling is supported 
//
// These are wrappers around the profiling backend:
//
//  PROFILING_FUNCTION
//     C++ only: used to mark the beginning of a profiled function
//
//  PROFILING_FUNCTION_BEGIN
//    to be used when entering a C/C++ function
//
//  PROFILING_FUNCTION_END
//    to be used when exiting a C/C++ function
//
//  PROFILING_REGION_BEGIN(name)
//    designates the beginning of a specific region for profiling (e.g. a 
//    for-loop)
//
//  PROFILING_REGION_END(name)
//    designates the end of a specific region for profiling
//
//  PROFILING_PHASE_BEGIN(name)
//    designates the beginning of a phase, creating a separate call-tree for 
//    the phase
//
//  PROFILING_PHASE_END(name)
//    designates the end of a phase
//
//  PROFILING_PAUSE
//    pauses profiling
//
//  PROFILING_RESUME
//    resumes profiling


#ifdef SCOREP_USER_ENABLE
# include <scorep/SCOREP_User.h>
///////////////////
// SCORE-P based 'manual' profiling
//
// SCOREP_USER_ENABLE is typically set when using the score-p compiler wrapper 
// with the --user argument

#  define PROFILING_FUNCTION_HEADER      \
    SCOREP_USER_REGION(SCOREP_USER_FUNCTION_NAME, \
                       SCOREP_USER_REGION_TYPE_FUNCTION)

// name -> name_handle
#  define PROFILING_NAME_TO_HANDLE(name) (name ## _handle)

#  define PROFILING_FUNCTION_BEGIN       SCOREP_USER_FUNC_BEGIN()
#  define PROFILING_FUNCTION_END         SCOREP_USER_FUNC_END()

#  define PROFILING_REGION_BEGIN(name)   \
    SCOREP_USER_REGION_DEFINE(PROFILING_NAME_TO_HANDLE(name)) ; \
    SCOREP_USER_REGION_BEGIN(PROFILING_NAME_TO_HANDLE(name), #name, \
                             SCOREP_USER_REGION_TYPE_COMMON)
#  define PROFILING_REGION_END(name) \
    SCOREP_USER_REGION_END(PROFILING_NAME_TO_HANDLE(name))

#  define PROFILING_PHASE_BEGIN(name)   \
    SCOREP_USER_REGION_DEFINE(PROFILING_NAME_TO_HANDLE(name)) ; \
    SCOREP_USER_REGION_BEGIN(PROFILING_NAME_TO_HANDLE(name), #name, \
                             SCOREP_USER_REGION_TYPE_PHASE)
#  define PROFILING_PHASE_END(name) \
    SCOREP_USER_REGION_END(PROFILING_NAME_TO_HANDLE(name))
#  define PROFILING_PAUSE SCOREP_RECORDING_OFF()
#  define PROFILING_RESUME SCOREP_RECORDING_ON()


#endif // SCOREP_USER_ENABLE


// Catch all cases where no profiling is defined/enabled
#if !defined(SCOREP_USER_ENABLE) && !defined(SOME_OTHER_PROFILING_LIB)
#  define PROFILING_FUNCTION_HEADER
#  define PROFILING_FUNCTION_BEGIN
#  define PROFILING_FUNCTION_END
#  define PROFILING_REGION_BEGIN(name)
#  define PROFILING_REGION_END(name)
#  define PROFILING_PHASE_BEGIN(name)
#  define PROFILING_PHASE_END(name)
#  define PROFILING_PAUSE
#  define PROFILING_RESUME
#endif // not SCOREP_USER_ENABLE && not SOME_OTHER_PROFILING_LIB


#ifdef SIMPLE_PROFILER

// Idea:
// LinAlg::SimpleProfiling -> has a vector for each function in 'abstract.h'  
// and possibly BLAS/*, LAPACK/* that adds the timing information
//
// Have an object that takes a name, at construction takes the time and at 
// destruction records it in the corresponding vector (which is lock guarded?)
//
// Provide
//
//  void show(const char* functionname, bool verbose = false)
//
//  void phase(const char* phasename) {
//
//  }
//
//
// Global Hash that maps "Function-name-string" -> vector & lock

namespace LinAlg {

namespace SimpleProfiler {

// Often used types
typedef const char*               key_t;
typedef std::pair<size_t, size_t> range_t;

// Global data structures, all 'defined' in profiling.cc

extern std::atomic<bool> simple_profiler_active;

// The global lock for all writing operations:
//  - registering a function to profiles
//  - registering a phase
extern Threads::Mutex simple_profiler_global_lock;

// key: __PRETTY_FUNCTION__ or some other unique function identifier
// value: corresponding function profile
struct FunctionProfile;
extern std::unordered_map<key_t, FunctionProfile> profiles; 
// TODO: make a compiler versioning check to ensure that the  
// __PRETTY_FUNCTION__ is supported by the invoked compiler

struct Phase;
extern std::vector<Phase> phases;


/** \brief            Container for the runtimes of a specific function
 */
struct FunctionProfile {

  key_t function_name;
  std::vector<size_t> times;    // unit: milliseconds

  // Lock has to be acquired before adding a time record to times
  Threads::Mutex lock;

  //FunctionProfile() = delete;   // no can do: unordered_map[] needs a zero
                                  // initializer as it might add an element
  FunctionProfile() : FunctionProfile("uninitialized") {};
  explicit FunctionProfile(key_t name);

  inline void add_record(size_t duration); // unit: milliseconds

};

/** \brief            Register runtime measurement
 *
 *  \param[in]        duration
 */
inline void FunctionProfile::add_record(size_t duration) {

  Threads::MutexLock my_lock(lock);

  times.push_back(duration);

}

/** \brief            Object that records time between its creation and its 
 *                    destruction
 *
 *  \param[in]        duration
 */
class TimeKeeper {

 public:
  TimeKeeper() = delete;
  TimeKeeper(key_t function_name);
  ~TimeKeeper();

 private:
  bool                                           keeping_time;
  std::chrono::high_resolution_clock::time_point call_begin;
  FunctionProfile*                               profile;

};

/** \brief            A segment of the total execution
 */
struct Phase {

  key_t                              phase_name;
  std::unordered_map<key_t, range_t> count_ranges; 
  bool                               closed;

  Phase() = delete;
  Phase(key_t name) : phase_name(name), closed(false) { }

};

// Initialization
#define SIMPLE_PROFILE_FIRST_PHASE_NAME "initial_phase__"

// Mark the begin of a new phase
void begin_phase(key_t phase_name);

// Wrap up a phase
void end_phase(key_t phase_name);

// A preprocessor macro that hijacks the PROFILING_FUNCTION_HEADER macro from 
// above and extends it
//
// Same for PROFILING_PAUSE and PROFILING_RESUME
#define SOMEMACRO

// Enable/disable the profiler
inline void pause()  { simple_profiler_active = false; }
inline void resume() { simple_profiler_active = true;  }

// Printing functions
void print(bool verbose = false);
void print_phase(Phase& phase, bool verbose = false);
void print_phase(key_t phase_name, bool verbose = false);
void print_function(key_t function_name, range_t range, bool verbose = false);
void print_function(key_t function_name, bool verbose);
void print_function(key_t function_name, key_t phase_name, bool verbose);
void print_debug();

} /* namespace LinAlg::SimpleProfiler */

} /* namespace LinAlg */

#endif /* SIMPLE_PROFILER */

#endif /* PROFILING_H_ */
