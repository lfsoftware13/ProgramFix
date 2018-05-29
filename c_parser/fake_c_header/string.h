
typedef long unsigned int size_t;


extern void *memcpy (void * __dest, const void * __src, size_t __n);


extern void *memmove (void *__dest, const void *__src, size_t __n);

extern void *memset (void *__s, int __c, size_t __n);


extern int memcmp (const void *__s1, const void *__s2, size_t __n);

extern void *memchr (const void *__s, int __c, size_t __n);


extern char *strcpy (char * __dest, const char * __src);

extern char *strncpy (char * __dest, const char * __src, size_t __n);


extern char *strcat (char * __dest, const char * __src);

extern char *strncat (char * __dest, const char * __src, size_t __n);


extern int strcmp (const char *__s1, const char *__s2);

extern int strncmp (const char *__s1, const char *__s2, size_t __n);


extern int strcoll (const char *__s1, const char *__s2);

extern size_t strxfrm (char * __dest, const char * __src, size_t __n);


extern char *strchr (const char *__s, int __c);

extern char *strrchr (const char *__s, int __c);

extern size_t strcspn (const char *__s, const char *__reject);


extern size_t strspn (const char *__s, const char *__accept);

extern char *strpbrk (const char *__s, const char *__accept);

extern char *strstr (const char *__haystack, const char *__needle);


extern char *strtok (char * __s, const char * __delim);


extern size_t strlen (const char *__s);




extern char *strerror (int __errnum);
