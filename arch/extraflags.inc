#####
# GCC
#
# All warnings are errors
GCC_EXTRA_FLAGS  = -Werror

# Turn on (most) warnings
GCC_EXTRA_FLAGS += -Wall

# Extra warnings not enabled by Wall
GCC_EXTRA_FLAGS += -Wextra -Wno-unused-parameter

# Ensure ISO compliant code
GCC_EXTRA_FLAGS += -Wpedantic

# Stop compilation at first error
#GCC_EXTRA_FLAGS += -Wfatal-errors

# others to consider
# 	-fmudflap/-fmudflapth			# Overflow preventing instrumentation


# vim: filetype=make:
