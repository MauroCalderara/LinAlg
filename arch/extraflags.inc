#####
# GCC
#
# All warnings are errors
GCC_EXTRA_FLAGS  = -Werror

# Turn on (most) warnings
GCC_EXTRA_FLAGS += -Wall

# Extra warnings not enabled by Wall
GCC_EXTRA_FLAGS += -Wextra

# Ensure ISO compliant code
GCC_EXTRA_FLAGS += -Wpedantic

# This is needed for our casting
GCC_EXTRA_FLAGS += -Wno-unused-parameter

# This is required by score-p
GCC_EXTRA_FLAGS += -Wno-unused-variable

# This is required in some uses of SubBlock w/ -O2
GCC_EXTRA_FLAGS += -Wno-strict-overflow

# Stop compilation at first error
GCC_EXTRA_FLAGS += -Wfatal-errors



#####
# CLANG
#
# All warnings are errors
CLANG_EXTRA_FLAGS  = -Werror

# Turn on all warnings
CLANG_EXTRA_FLAGS += -Weverything

# Ensure ISO compliant code
CLANG_EXTRA_FLAGS += -pedantic

# Non-ISO is an error
CLANG_EXTRA_FLAGS += -pedantic-errors

# Be verbose about the error category
CLANG_EXTRA_FLAGS += -fdiagnostics-show-category=name

# Have nice template mismatch output
CLANG_EXTRA_FLAGS += -fdiagnostics-show-template-tree

# This is needed for our casting
CLANG_EXTRA_FLAGS += -Wno-unused-parameter

# This is required by score-p
CLANG_EXTRA_FLAGS += -Wno-unused-variable

# This is required in some uses of SubBlock w/ -O2
CLANG_EXTRA_FLAGS += -Wno-strict-overflow

# Check doxygen comments
CLANG_EXTRA_FLAGS += -Wdocumentation

# Have the optimizer tell what is done
CLANG_EXTRA_FLAGS += -Rpass=.*

# ... what isn't
CLANG_EXTRA_FLAGS += -Rpass-missed=.*

# ... and why it decides like that
CLANG_EXTRA_FLAGS += -Rpass-analysis=.*

# Flags for the instrumenting profiler
# 
# Usage: see clang documentation, but here is a short howto:
# 	> make profile_generator				    # build the instrumented binary
#	> LLVM_PROFILE_FILE="code-%p.profraw"		# set profiling output filename
#	> ./my_binary some_representative_input		# run binary with real input
#	> ...										# optional: run other inputs
#	> llvm-profdata merge -output=code.profdata code-*.profraw
#												# merge/convert profiles
#	> make profile_optimized					# make a new, optimized binary
CLANG_CREATE_PROFIE_FLAGS = -fprofile-instr-generate
CLANG_USE_PROFILE_FLAGS   = -fprofile-instr-use=code.profdata


# vim: filetype=make:
