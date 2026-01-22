#error "SSE2 instruction set not enabled"
#endif

#include <xmmintrin.h>

typedef double __m128d __attribute__((__vector_size__(16)));
