/** \file
 *
 *  \brief            A timer class
 *
 *  \date             Created:  Aug 10, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mauro@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_UTILITIES_TIMER_H_
#define LINALG_UTILITIES_TIMER_H_

#include <chrono>     // std::chrono and all submembers
#include <cstdio>     // std::fprint
#include <string>

#include "stringformat.h"

namespace LinAlg {

namespace Utilities {


/** \brief            A time measurement facility
 */
struct Timer {

#ifndef DOXYGEN_SKIP
  typedef std::chrono::high_resolution_clock::time_point time_point_t;

  time_point_t start;
  time_point_t stop;
  std::chrono::duration<double> time_span;
  std::string name;
#endif

  /** \brief          Constructor
   *
   *  \param[in]      us
   *                  OPTIONAL: Whether to print microsecond resolution. If
   *                  true, the elapsed time will be printed in units of
   *                  microseconds, otherwise in units of seconds. DEFAULT:
   *                  false.
   */
  Timer(bool us = false)
      : start(std::chrono::high_resolution_clock::now()),
        name("") { }
  /** \brief          Constructor with a name string
   *
   *  \param[in]      name_in
   *                  Name that will be printed before the timer's output
   *
   *  \param[in]      us
   *                  OPTIONAL: Whether to print microsecond resolution. If
   *                  true, the elapsed time will be printed in units of
   *                  microseconds, otherwise in units of seconds. DEFAULT:
   *                  false.
   */
  template <typename... Us>
  Timer(const char* name_in, Us... formatargs)
      : start(std::chrono::high_resolution_clock::now()),
        name(stringformat(name_in, formatargs...)) { }

  /** \brief          Start the timer
   */
  inline void set() { start = std::chrono::high_resolution_clock::now(); }

  /** \brief          Stop the timer
   *
   *  \returns        The time in seconds or microseconds (depending on the
   *                  parameter given at construction time)
   */
  inline double measure() {

    using std::chrono::duration_cast;
    using std::chrono::duration;

    stop = std::chrono::high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(stop - start);

    return time_span.count();

  }

  /** \brief          Start the timer (MATLAB&reg; alike)
   */
  inline void tic() { set(); }

  /** \brief          Stop the timer, print the elapsed time since the last
   *                  set() or tic() (MATLAB&reg; alike)
   *
   *  \returns        The time in seconds or microseconds (depending on the
   *                  parameter given at construction time)
   */
  inline double toc() {

    auto tmp = measure();
    std::printf("%s : %fs\n", name.c_str(), tmp);

    return tmp;

  }

};

struct HiResTimer {

#ifndef DOXYGEN_SKIP
  typedef std::chrono::high_resolution_clock::time_point time_point_t;

  time_point_t start;
  time_point_t stop;
  std::chrono::duration<double> time_span;
  std::string name;
#endif

  /** \brief          Constructor
   */
  HiResTimer()
           : start(std::chrono::high_resolution_clock::now()),
             name("") {}
  /** \brief          Constructor with a name string
   *
   *  \param[in]      name_in
   *                  Name that will be printed before the timer's output
   *
   *  \param[in]      formatargs
   *                  OPTIONAL: printf style format arguments
   */
  template <typename... Us>
  HiResTimer(const char* name_in, Us... formatargs)
           : start(std::chrono::high_resolution_clock::now()),
             name(stringformat(name_in, formatargs...)) {}

  /** \brief          Start the timer
   */
  inline void set() { start = std::chrono::high_resolution_clock::now(); }

  /** \brief          Stop the timer
   *
   *  \returns        The time in seconds or microseconds (depending on the
   *                  parameter given at construction time)
   */
  inline double measure() {

    using std::chrono::duration_cast;
    using std::chrono::duration;

    stop = std::chrono::high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(stop - start);

    return time_span.count() * 1000000;

  }

  /** \brief          Start the timer (MATLAB&reg; alike)
   */
  inline void tic() { set(); }

  /** \brief          Stop the timer, print the elapsed time since the last
   *                  set() or tic() (MATLAB&reg; alike)
   *
   *  \returns        The time in seconds or microseconds (depending on the
   *                  parameter given at construction time)
   */
  inline double toc() {

    auto tmp = measure();
    std::printf("%s : %fus\n", name.c_str(), tmp);

    return tmp;

  }

};

} /* namespace LinAlg::Utilities */

} /* namespace LinAlg */


#endif /* LINALG_UTILITIES_TIMER_H_ */
