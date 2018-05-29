#include<stddef.h>

typedef int wchar_t;

typedef struct
  {
    int quot;
    int rem;
  } div_t;

typedef struct
  {
    long int quot;
    long int rem;
  } ldiv_t;

extern double atof (const char *__nptr);

extern int atoi (const char *__nptr);

extern long int atol (const char *__nptr);

extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr);

extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base);

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base);

extern void *calloc (size_t __nmemb, size_t __size);

extern void free (void *__ptr);

extern void *malloc (size_t __size);

extern void *realloc (void *__ptr, size_t __size);

extern void abort (void) ;

extern int atexit (void (*__func) (void));

extern void exit (int __status);

extern char *getenv (const char *__name);

extern int system (const char *__command) ;

extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar);

extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar);

extern int abs (int __x);

extern div_t div (int __numer, int __denom);

extern long int labs (long int __x);

extern ldiv_t ldiv (long int __numer, long int __denom);

extern int rand (void);

extern void srand (unsigned int __seed);

extern int mblen (const char *__s, size_t __n);

extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n);

extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n);

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n);

extern int wctomb (char *__s, wchar_t __wchar);

