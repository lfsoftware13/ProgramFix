typedef float float_t;
typedef double double_t;

extern double acos (double __x);

extern double asin (double __x);

extern double atan (double __x);

extern double atan2 (double __y, double __x);

 extern double cos (double __x);

 extern double sin (double __x);

 extern double tan (double __x);

 extern double cosh (double __x);

extern double sinh (double __x);

extern double tanh (double __x);

extern double acosh (double __x);

extern double asinh (double __x);

extern double atanh (double __x);

 extern double exp (double __x);

extern double frexp (double __x, int *__exponent);

extern double ldexp (double __x, int __exponent);

 extern double log (double __x);

extern double log10 (double __x);

extern double modf (double __x, double *__iptr);

extern double expm1 (double __x);

extern double log1p (double __x);

extern double logb (double __x);

extern double exp2 (double __x);

extern double log2 (double __x);

 extern double pow (double __x, double __y);

extern double sqrt (double __x);

extern double hypot (double __x, double __y);

extern double cbrt (double __x);

extern double ceil (double __x);

extern double fabs (double __x);

extern double floor (double __x);

extern double fmod (double __x, double __y);

extern double copysign (double __x, double __y);

extern double nan (const char *__tagb);

extern double erf (double);

extern double erfc (double);

extern double lgamma (double);

extern double tgamma (double);

extern double rint (double __x);

extern double nextafter (double __x, double __y);

extern double nexttoward (double __x, long double __y);

extern double remainder (double __x, double __y);

extern double scalbn (double __x, int __n);

extern int ilogb (double __x);

extern double scalbln (double __x, long int __n);

extern double nearbyint (double __x);

extern double round (double __x);

extern double trunc (double __x);

extern double remquo (double __x, double __y, int *__quo);

extern long int lrint (double __x);

extern long long int llrint (double __x);

extern long int lround (double __x);

extern long long int llround (double __x);

extern double fdim (double __x, double __y);

extern double fmax (double __x, double __y);

extern double fmin (double __x, double __y);

extern double fma (double __x, double __y, double __z);

extern float acosf (float __x);

extern float asinf (float __x);

extern float atanf (float __x);

extern float atan2f (float __y, float __x);

 extern float cosf (float __x);

 extern float sinf (float __x);

extern float tanf (float __x);

extern float coshf (float __x);

extern float sinhf (float __x);

extern float tanhf (float __x);

extern float acoshf (float __x);

extern float asinhf (float __x);

extern float atanhf (float __x);

 extern float expf (float __x);

extern float frexpf (float __x, int *__exponent);

extern float ldexpf (float __x, int __exponent);

 extern float logf (float __x);

extern float log10f (float __x);

extern float modff (float __x, float *__iptr);

extern float expm1f (float __x);

extern float log1pf (float __x);

extern float logbf (float __x);

extern float exp2f (float __x);

extern float log2f (float __x);

 extern float powf (float __x, float __y);

extern float sqrtf (float __x);

extern float hypotf (float __x, float __y);

extern float cbrtf (float __x);

extern float ceilf (float __x);

extern float fabsf (float __x);

extern float floorf (float __x);

extern float fmodf (float __x, float __y);

extern float copysignf (float __x, float __y);

extern float nanf (const char *__tagb);

extern float erff (float);

extern float erfcf (float);

extern float lgammaf (float);

extern float tgammaf (float);

extern float rintf (float __x);

extern float nextafterf (float __x, float __y);

extern float nexttowardf (float __x, long double __y);

extern float remainderf (float __x, float __y);

extern float scalbnf (float __x, int __n);

extern int ilogbf (float __x);

extern float scalblnf (float __x, long int __n);

extern float nearbyintf (float __x);

extern float roundf (float __x);

extern float truncf (float __x);

extern float remquof (float __x, float __y, int *__quo);

extern long int lrintf (float __x);

extern long long int llrintf (float __x);

extern long int lroundf (float __x);

extern long long int llroundf (float __x);

extern float fdimf (float __x, float __y);

extern float fmaxf (float __x, float __y);

extern float fminf (float __x, float __y);

extern float fmaf (float __x, float __y, float __z);

extern long double acosl (long double __x);

extern long double asinl (long double __x);

extern long double atanl (long double __x);

extern long double atan2l (long double __y, long double __x);

 extern long double cosl (long double __x);

 extern long double sinl (long double __x);

extern long double tanl (long double __x);

extern long double coshl (long double __x);

extern long double sinhl (long double __x);

extern long double tanhl (long double __x);

extern long double acoshl (long double __x);

extern long double asinhl (long double __x);

extern long double atanhl (long double __x);

 extern long double expl (long double __x);

extern long double frexpl (long double __x, int *__exponent);

 extern long double logl (long double __x);

extern long double log10l (long double __x);

extern long double modfl (long double __x, long double *__iptr);

extern long double expm1l (long double __x);

extern long double log1pl (long double __x);

extern long double logbl (long double __x);

extern long double exp2l (long double __x);

extern long double log2l (long double __x);

 extern long double powl (long double __x, long double __y);

extern long double sqrtl (long double __x);

extern long double hypotl (long double __x, long double __y);

extern long double cbrtl (long double __x);

extern long double ceill (long double __x);

extern long double fabsl (long double __x);

extern long double floorl (long double __x);

extern long double fmodl (long double __x, long double __y);

extern long double copysignl (long double __x, long double __y);

extern long double nanl (const char *__tagb);

extern long double erfl (long double);

extern long double erfcl (long double);

extern long double lgammal (long double);

extern long double tgammal (long double);

extern long double rintl (long double __x);

extern long double nextafterl (long double __x, long double __y);

extern long double nexttowardl (long double __x, long double __y);

extern long double remainderl (long double __x, long double __y);

extern long double scalbnl (long double __x, int __n);

extern int ilogbl (long double __x);

extern long double scalblnl (long double __x, long int __n);

extern long double nearbyintl (long double __x);

extern long double roundl (long double __x);

extern long double truncl (long double __x);

extern long double remquol (long double __x, long double __y, int *__quo);

extern long int lrintl (long double __x);

extern long long int llrintl (long double __x);

extern long int lroundl (long double __x);

extern long long int llroundl (long double __x);

extern long double fdiml (long double __x, long double __y);

extern long double fmaxl (long double __x, long double __y);

extern long double fminl (long double __x, long double __y);

extern long double fmal (long double __x, long double __y, long double __z);




