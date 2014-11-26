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


// Currently only score-p based profiling is supported
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
#include <scorep/SCOREP_User.h>
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


#endif /* PROFILING_H_ */
