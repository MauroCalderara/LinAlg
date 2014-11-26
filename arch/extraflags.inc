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
#GCC_EXTRA_FLAGS += -Wfatal-errors

# others to consider
# 	-fmudflap/-fmudflapth			# Overflow preventing instrumentation


# vim: filetype=make:
