/** \file
 *
 *  \brief            Code for the simple profiling functionality
 *
 *  \date             Created:  Nov 30, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */

#include "preprocessor.h"

#ifdef SIMPLE_PROFILER
# include <atomic>
# include <unordered_map>
# include <functional>
# include <vector>
# include <chrono>
# include <cstdio>
# include <cmath>
# include <utility>
# include <cassert>
# ifdef USE_POSIX_THREADS
#   include <pthread.h>
# else      // use C++11 primitives
#   include <mutex>
# endif 

# include "exceptions.h"
# include "profiling.h"
# include "threads.h"

namespace LinAlg {

namespace SimpleProfiler {

// The global structures shared by all threads (defined extern in profiling.h)
std::atomic<bool>                          simple_profiler_active(true);
Threads::Mutex                             simple_profiler_global_lock;
std::unordered_map<key_t, FunctionProfile> profiles;
std::vector<Phase>                         phases;

// FunctionProfile
//
/** \brief            Constructor from name
 *
 *  \param[in]        name
 *                    Name of the function for which this is the profile.  
 *                    Typically the macro __PRETTY_FUNCTION__ or some other 
 *                    unique identifier for the function
 */
FunctionProfile::FunctionProfile(key_t name) : function_name(name) {

  times.reserve(SIMPLE_PROFILER_PREALLOCATED_RECORDS);

}


// TimeKeeper
//
/** \brief            Constructor from name
 *
 *  \param[in]        name
 *                    Name of the function for which this is the profile.  
 *                    Typically the macro __PRETTY_FUNCTION__ or some other 
 *                    unique identifier for the function
 */
TimeKeeper::TimeKeeper(key_t function_name) : keeping_time(true) {

  if (!simple_profiler_active) {
    keeping_time = false;
    return;
  }

  // Check if there is a default phase
  if (phases.size() == 0) {
    Threads::MutexLock  my_lock(simple_profiler_global_lock);
    phases.emplace_back(SIMPLE_PROFILE_FIRST_PHASE_NAME);
  }

  // Check if we need to register a new profile
  if (profiles.count(function_name) == 0) {
  
    Threads::MutexLock  my_lock(simple_profiler_global_lock);
  
    // In the meantime somebody could have added the profile so we check again
    // before adding
    if (profiles.count(function_name) == 0) {
    
      profiles.emplace(function_name, FunctionProfile(function_name));
    
    }
  
  }

  profile = &(profiles[function_name]);
  call_begin = std::chrono::high_resolution_clock::now();

}

/** \brief            Destructor, records time of the function call, 
 *                    automatically executed when the function returns
 */
TimeKeeper::~TimeKeeper() {

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::milliseconds;

  if (!keeping_time) return;

  auto call_end = high_resolution_clock::now();
  size_t duration = duration_cast<milliseconds>(call_end - call_begin).count();
  profile->add_record(duration);

}

// Phase
//
/** \brief            Record the begin of a new phase
 *
 *  \param[in]        name
 *                    Name of the phase
 *
 *  \note             It is not neccessary to close a phase before beginning a 
 *                    new one. The previous phase will be closed automatically.
 */
void begin_phase(key_t name) {

  // We record a new phase by 'closing off' the old one. Since at the beginning 
  // of a phase we don't know what routines will register, we have to do the 
  // accounting ex post when a new phase begins

  // Check if there is a default phase
  if (phases.size() == 0) {
    Threads::MutexLock  my_lock(simple_profiler_global_lock);
    phases.emplace_back(SIMPLE_PROFILE_FIRST_PHASE_NAME);
  }

  // Close off the previous phase
  auto last_phase = phases.back();
  end_phase(phases.back().phase_name);
  last_phase.closed = true;

  Threads::MutexLock  my_lock(simple_profiler_global_lock);

  phases.emplace_back(name);
  auto current_phase = phases.back();

  for (auto &key_val_pair: profiles) {

    auto profile = &(key_val_pair.second);

    Threads::MutexLock my_profile_lock(profile->lock);

    auto function_name = profile->function_name;
    auto current_count = profile->times.size();

    // Register the counts for the current phase
    current_phase.count_ranges.emplace(function_name, 
                                            range_t(current_count, 0));

  }

}

/** \brief            End/close a phase
 *
 *  \param[in]        phase
 *                    The phase to close off (it is assumed that the phase is 
 *                    the last phase that was created)
 */
void end_phase(Phase& phase) {

  for (auto &key_val_pair: profiles) {

    auto profile = &(key_val_pair.second);

    Threads::MutexLock my_profile_lock(profile->lock);

    auto function_name = profile->function_name;
    auto current_count = profile->times.size();

    // Ex-post book keeping
    auto last_profiles_count = phase.count_ranges.count(function_name);
    if (last_profiles_count == 0) {
      
      // Newly registered function
      phase.count_ranges.emplace(function_name, range_t(0, current_count));
    
    } else if (last_profiles_count == 1) {
    
      // Function already registered at beginning of last function
      auto start_count = phase.count_ranges[function_name].first;
      phase.count_ranges[function_name] = range_t(start_count, current_count);
    
    } else {

      assert(false);

    }

  }

}

/** \brief            End/close a phase
 *
 *  \param[in]        name
 *                    Name of the phase to close off (it is assumed that the 
 *                    phase is the last phase that was created)
 */
void end_phase(key_t phase_name) {

  // Check if there is a default phase
  if (phases.size() == 0) {
    Threads::MutexLock  my_lock(simple_profiler_global_lock);
    phases.emplace_back(SIMPLE_PROFILE_FIRST_PHASE_NAME);
  }

  std::string lookup_name(phase_name);
  std::string element_name;

  for (size_t i = 0; i < phases.size(); ++i) {
  
    element_name = phases[i].phase_name;

    if (lookup_name == element_name) {

      end_phase(phases[i]);
      return;

    }

  }

}

// Printing of the measurements
//
/** \brief            Print all profiling records
 *
 *  \param[in]        verbose
 *                    Optional: Verbose output (i.e. every single record in 
 *                    addition to the aggregate data). Default = false
 */
void print(bool verbose) {

  for (auto& phase : phases) print_phase(phase, verbose);

}

/** \brief            Print all profiling records in one phase
 *
 *  \param[in]        phase
 *                    Phase or name of the phase (as created with begin_phase())
 *
 *  \param[in]        verbose
 *                    Optional: Verbose output (i.e. every single record in 
 *                    addition to the aggregate data). Default = false
 */
void print_phase(Phase& phase, bool verbose) {

  if (!phase.closed) end_phase(phase);

  if (std::string(phase.phase_name) != 
      std::string(SIMPLE_PROFILE_FIRST_PHASE_NAME)) {
    printf(" %s\n", phase.phase_name);
  }

  for (auto &entry : phase.count_ranges) {

    auto function_name = entry.first;
    auto range = entry.second;

    print_function(function_name, range, verbose);

  }

}
/** \overload
 */
void print_phase(key_t phase_name, bool verbose) {

  std::string lookup_name(phase_name);
  std::string element_name;

  for (size_t i = 0; i < phases.size(); ++i) {
  
    element_name = phases[i].phase_name;

    if (lookup_name == element_name) {

      print_phase(phases[i], verbose);
      return;

    }

  }

  throw excBadArgument("print_phase(): phase not found");

}

/** \brief            Print the records for a given function id and range
 *
 *  \param[in]        function_name
 *                    Name of the function to print
 *
 *  \param[in]        range
 *                    Range of measurements to print. Upper limit is exclusive.
 *
 *  \param[in]        verbose
 *                    Optional: whether to print all measurements or only the 
 *                    aggregate data (count/min/max/avg/stddev). Default: false.
 */
void print_function(key_t function_name, range_t range,
                    bool verbose) {

  auto  profile = &(profiles[function_name]);
  auto  start = range.first;
  auto  stop  = (range.second == 0) ? profile->times.size() : range.second;

  if (start == stop) return;

  size_t min = std::numeric_limits<size_t>::max();
  size_t max = 0;
  size_t sum = 0;
  size_t square_sum = 0;

  printf("  %30s", function_name);
  if (verbose) printf("\n");

  for (size_t i = start; i < stop; ++i) {
    
    auto record = profile->times[i];

    min         = (record < min) ? record : min;
    max         = (record > max) ? record : max;
    sum        += record;
    square_sum += record * record;
  
    if (verbose) printf("   %5zu: %f s\n", i, record/1000.0);
  
  }

  size_t count   = stop - start;
  double average = sum / count;
  double stddev  = std::sqrt(square_sum / count - average * average);

  if (verbose) printf("\n    ------------------------\n");
  printf("    %f / %f / %f / %f / %f s\n",
         count/1000.0, min/1000.0, max/1000.0, average/1000.0, stddev/1000.0);

}
/** \overload
 */
void print_function(key_t function_name, bool verbose) {

  auto last = profiles[function_name].times.size();
  range_t range(0, last);

  print_function(function_name, range, verbose);

}
/** \brief            Print a function's runtimes during one phase
 *
 *  \param[in]        function_name
 *
 *  \param[in]        phase_name
 *
 *  \param[in]        verbose
 */
void print_function(key_t function_name, key_t phase_name, bool verbose) {

  std::string lookup_name(phase_name);
  std::string element_name;
  range_t     range;

  for (size_t i = 0; i < phases.size(); ++i) {
  
    element_name = phases[i].phase_name;

    if (lookup_name == element_name) {

      if (!phases[i].closed) end_phase(phases[i]);

      auto range = phases[i].count_ranges[function_name];

      printf(" %s\n", phase_name);
      print_function(function_name, range, verbose);

      return;

    }

  }

  throw excBadArgument("print_function(): phase not found");

}

/** \brief            Display the contents of all profiling related structures
 */
void print_debug() {

  printf("simple_profiler_active = %s\n", (simple_profiler_active) ? "true" :
                                                                     "false");

  printf("profiles (%zu total):\n", profiles.size());
  for (auto& profile : profiles) {
    printf("  %s (%zu records)\n", profile.second.function_name,
           profile.second.times.size());
  }

  printf("phases (%zu total):\n", phases.size());
  for (auto& phase : phases) {

    if (!phase.closed) end_phase(phase);

    printf("  %s, %zu count ranges:\n", phase.phase_name, 
           phase.count_ranges.size());

    for (auto& count_range : phase.count_ranges) {
      printf("    %s: %zu:%zu\n", count_range.first, count_range.second.first,
             count_range.second.second);
    }

  }

}

}

}

#endif
