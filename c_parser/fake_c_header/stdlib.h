typedef long unsigned int size_t;

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



typedef struct
  {
    long long int quot;
    long long int rem;
  } lldiv_t;


extern double atof (const char *__nptr);

extern int atoi (const char *__nptr);

extern long int atol (const char *__nptr);



extern long long int atoll (const char *__nptr);


extern double strtod (const char * __nptr, char ** __endptr);


extern float strtof (const char * __nptr, char ** __endptr);

extern long double strtold (const char * __nptr, char ** __endptr);


extern long int strtol (const char * __nptr, char ** __endptr, int __base);

extern unsigned long int strtoul (const char * __nptr, char ** __endptr, int __base);

extern long long int strtoll (const char * __nptr, char ** __endptr, int __base);

extern unsigned long long int strtoull (const char * __nptr, char ** __endptr, int __base);

extern int rand (void);

extern void srand (unsigned int __seed);

extern void *malloc (size_t __size);

extern void *calloc (size_t __nmemb, size_t __size);


extern void *realloc (void *__ptr, size_t __size);

extern void free (void *__ptr);

extern void abort (void);


extern int atexit (void (*__func) (void));


extern void exit (int __status);


extern char *getenv (const char *__name);

extern int system (const char *__command) ;


typedef int (*__compar_fn_t) (const void *, const void *);


extern void *bsearch (const void *__key, const void *__base, size_t __nmemb, size_t __size, __compar_fn_t __compar);


extern void qsort (void *__base, size_t __nmemb, size_t __size, __compar_fn_t __compar);

extern int abs (int __x);
extern long int labs (long int __x);



extern long long int llabs (long long int __x);







extern div_t div (int __numer, int __denom);
extern ldiv_t ldiv (long int __numer, long int __denom);




extern lldiv_t lldiv (long long int __numer, long long int __denom);

extern int mblen (const char *__s, size_t __n);


extern int mbtowc (wchar_t * __pwc, const char * __s, size_t __n);


extern int wctomb (char *__s, wchar_t __wchar);

extern size_t mbstowcs (wchar_t * __pwcs, const char * __s, size_t __n);

extern size_t wcstombs (char * __s, const wchar_t * __pwcs, size_t __n);