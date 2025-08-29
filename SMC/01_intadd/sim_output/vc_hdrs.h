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


 extern void add32_128bit(/* INPUT */unsigned long long src0_high, /* INPUT */unsigned long long src0_low, /* INPUT */unsigned long long src1_high, /* INPUT */unsigned long long src1_low, /* INPUT */int sign_s0, /* INPUT */int sign_s1, /* OUTPUT */unsigned long long *dst_high, /* OUTPUT */unsigned long long *dst_low, /* OUTPUT */unsigned long long *st_high, /* OUTPUT */unsigned long long *st_low
);

 extern void add8_128bit(/* INPUT */unsigned long long src0_high, /* INPUT */unsigned long long src0_low, /* INPUT */unsigned long long src1_high, /* INPUT */unsigned long long src1_low, /* INPUT */unsigned long long src2_high, /* INPUT */unsigned long long src2_low, /* INPUT */int sign_s0, /* INPUT */int sign_s1, /* INPUT */int sign_s2, /* OUTPUT */unsigned long long *dst0_high, 
/* OUTPUT */unsigned long long *dst0_low, /* OUTPUT */unsigned long long *dst1_high, /* OUTPUT */unsigned long long *dst1_low, /* OUTPUT */unsigned long long *st_high, /* OUTPUT */unsigned long long *st_low);

#ifdef __cplusplus
}
#endif

