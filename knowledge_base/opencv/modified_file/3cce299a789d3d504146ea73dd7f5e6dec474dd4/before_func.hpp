    #if defined(__clang__)
      #define CV__FASTMATH_ENABLE_CLANG_MATH_BUILTINS
      #if !defined(CV_INLINE_ISNAN_DBL) && __has_builtin(__builtin_isnan)
        #define CV_INLINE_ISNAN_DBL(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISNAN_FLT) && __has_builtin(__builtin_isnan)
        #define CV_INLINE_ISNAN_FLT(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISINF_DBL) && __has_builtin(__builtin_isinf)
        #define CV_INLINE_ISINF_DBL(value) return __builtin_isinf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_FLT) && __has_builtin(__builtin_isinf)
        #define CV_INLINE_ISINF_FLT(value) return __builtin_isinf(value);
      #endif
    #elif defined(__GNUC__)
      #define CV__FASTMATH_ENABLE_GCC_MATH_BUILTINS
      #if !defined(CV_INLINE_ISNAN_DBL)
        #define CV_INLINE_ISNAN_DBL(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISNAN_FLT)
        #define CV_INLINE_ISNAN_FLT(value) return __builtin_isnanf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_DBL)
