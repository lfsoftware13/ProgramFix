

typedef long unsigned int size_t;


typedef long int __off_t;
typedef long int __off64_t;


typedef long int __ssize_t;


struct _IO_FILE;



typedef struct _IO_FILE FILE;


typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;

typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;

typedef void __gnuc_va_list;

typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;

};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};

struct _IO_FILE {
  int _flags;




  char* _IO_read_ptr;
  char* _IO_read_end;
  char* _IO_read_base;
  char* _IO_write_base;
  char* _IO_write_ptr;
  char* _IO_write_end;
  char* _IO_buf_base;
  char* _IO_buf_end;

  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;



  int _flags2;

  __off_t _old_offset;



  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];



  _IO_lock_t *_lock;

  __off64_t _offset;

  void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;

  size_t __pad5;
  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];

};


typedef struct _IO_FILE _IO_FILE;



typedef _G_fpos_t fpos_t;

extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;

extern int remove (const char *__filename);

extern int rename (const char *__old, const char *__new);

extern FILE *tmpfile (void);

extern char *tmpnam (char *__s);

extern int fclose (FILE *__stream);

extern int fflush (FILE *__stream);

extern FILE *fopen (const char * __filename, const char * __modes);

extern FILE *freopen (const char * __filename, const char * __modes, FILE * __stream) ;

extern void setbuf (FILE * __stream, char * __buf);

extern int setvbuf (FILE * __stream, char * __buf, int __modes, size_t __n);

extern int fprintf (FILE * __stream, const char * __format, ...);

extern int printf (const char * __format, ...);

extern int sprintf (char * __s, const char * __format, ...);

extern int vfprintf (FILE * __s, const char * __format, __gnuc_va_list __arg);

extern int vprintf (const char * __format, __gnuc_va_list __arg);

extern int vsprintf (char * __s, const char * __format, __gnuc_va_list __arg);

extern int snprintf (char * __s, size_t __maxlen, const char * __format, ...);

extern int vsnprintf (char * __s, size_t __maxlen, const char * __format, __gnuc_va_list __arg);

extern int fscanf (FILE * __stream, const char * __format, ...);

extern int scanf (const char * __format, ...);

extern int sscanf (const char * __s, const char * __format, ...);

extern int fscanf (FILE * __stream, const char * __format, ...);

extern int scanf (const char * __format, ...);

extern int sscanf (const char * __s, const char * __format, ...);

extern int vfscanf (FILE * __s, const char * __format, __gnuc_va_list __arg);

extern int vscanf (const char * __format, __gnuc_va_list __arg);

extern int vsscanf (const char * __s, const char * __format, __gnuc_va_list __arg);

extern int vfscanf (FILE * __s, const char * __format, __gnuc_va_list __arg);

extern int vscanf (const char * __format, __gnuc_va_list __arg);

extern int vsscanf (const char * __s, const char * __format, __gnuc_va_list __arg);

extern int fgetc (FILE *__stream);

extern int getc (FILE *__stream);

extern int getchar (void);

extern int fputc (int __c, FILE *__stream);

extern int putc (int __c, FILE *__stream);

extern int putchar (int __c);

extern char *fgets (char * __s, int __n, FILE * __stream);

extern char *gets (char *__s);

extern int fputs (const char * __s, FILE * __stream);

extern int puts (const char *__s);

extern int ungetc (int __c, FILE *__stream);

extern size_t fread (void * __ptr, size_t __size, size_t __n, FILE * __stream);

extern size_t fwrite (const void * __ptr, size_t __size, size_t __n, FILE * __s);

extern int fseek (FILE *__stream, long int __off, int __whence);

extern long int ftell (FILE *__stream);

extern void rewind (FILE *__stream);

extern int fgetpos (FILE * __stream, fpos_t * __pos);

extern int fsetpos (FILE *__stream, const fpos_t *__pos);

extern void clearerr (FILE *__stream);

extern int feof (FILE *__stream);

extern int ferror (FILE *__stream);

extern void perror (const char *__s);

