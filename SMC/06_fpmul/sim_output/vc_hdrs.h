#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include "svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _VC_TYPES_
#define _VC_TYPES_
/* common definitions shared with DirectC.h */

typedef unsigned int U;
typedef unsigned char UB;
typedef unsigned char scalar;
typedef struct { U c; U d;} vec32;

#define scalar_0 0
#define scalar_1 1
#define scalar_z 2
#define scalar_x 3

extern long long int ConvUP2LLI(U* a);
extern void ConvLLI2UP(long long int a1, U* a2);
extern long long int GetLLIresult();
extern void StoreLLIresult(const unsigned int* data);
typedef struct VeriC_Descriptor *vc_handle;

#ifndef SV_3_COMPATIBILITY
#define SV_STRING const char*
#else
#define SV_STRING char*
#endif

#endif /* _VC_TYPES_ */


 extern unsigned short int dpi_f16_mul(/* INPUT */unsigned short int a, /* INPUT */unsigned short int b);

 extern unsigned int dpi_f32_mul(/* INPUT */unsigned int a, /* INPUT */unsigned int b);

 extern unsigned int dpi_get_inexact_flag();

 extern unsigned int dpi_get_underflow_flag();

 extern unsigned int dpi_get_overflow_flag();

 extern unsigned int dpi_get_infinite_flag();

 extern unsigned int dpi_get_invalid_flag();

 extern unsigned int dpi_get_exception_flags();

 extern void dpi_clear_exception_flags();

#ifdef __cplusplus
}
#endif

