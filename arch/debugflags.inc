#####
# GCC
#
# Debug symbols
GCC_DEBUG_FLAGS  = -g

# Enable array bounds checking
GCC_DEBUG_FLAGS += -fsanitize=address

# Have nicer stack traces for the above
GCC_DEBUG_FLAGS += -fno-omit-frame-pointer


##############
# LLVM / clang
#
# Debug symbols
CLANG_DEBUG_FLAGS  = -g

# Turn on all of clang's non-conflicting sanitizers
CLANG_DEBUG_FLAGS += -fsanitize=integer,undefined,dataflow,unsigned-integer-overflow

# Have nicer stack traces
CLANG_DEBUG_FLAGS += -fno-omit-frame-pointer

# Flags for the address sanitizer
CLANG_DEBUG_FLAGS_ADDRESS  = -fsanitize=address

# Flags for the thread sanitizer
CLANG_DEBUG_FLAGS_THREAD   = -fsanitize=thread

# Flags for the memory sanitizer
CLANG_DEBUG_FLAGS_MEMORY   = -fsanitize=memory

# Have the memory sanitizer trace origin of uninitialized memory
CLANG_DEBUG_FLAGS_MEMORY  += -fsanitize-memory-track-origins=2

# vim: filetype=make:
