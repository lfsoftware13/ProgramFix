#include<stddef.h>

struct _FILE{

};

typedef struct _FILE FILE;

typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef _G_fpos_t fpos_t;

extern int fclose (FILE *__stream);

extern void clearerr (FILE *__stream);

extern int feof (FILE *__stream);

extern int ferror (FILE *__stream);

extern int fflush (FILE *__stream);

extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);

extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) ;

extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;

extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) ;

extern int fseek (FILE *__stream, long int __off, int __whence);

extern int fsetpos (FILE *__stream, const fpos_t *__pos);

extern long int ftell (FILE *__stream) ;

extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);

extern int remove (const char *__filename);

extern int rename (const char *__old, const char *__new);

extern void rewind (FILE *__stream);

extern void setbuf (FILE *__restrict __stream, char *__restrict __buf);

int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n);

extern FILE *tmpfile (void) ;

extern char *tmpnam (char *__s) ;

extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);

extern int printf (const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...);

extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) ;

extern int scanf (const char *__restrict __format, ...) ;

extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...);

extern int fgetc (FILE *__stream);

extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream);

extern int fputc (int __c, FILE *__stream);

extern int fputs (const char *__restrict __s, FILE *__restrict __stream);

extern int getc (FILE *__stream);

extern char *gets (char *__s);

extern int putc (int __c, FILE *__stream);

extern int putchar (int __c);

extern int puts (const char *__s);

extern int ungetc (int __c, FILE *__stream);

extern void perror (const char *__s);

