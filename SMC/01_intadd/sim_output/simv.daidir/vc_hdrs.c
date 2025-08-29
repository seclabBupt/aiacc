#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include "svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* VCS error reporting routine */
extern void vcsMsgReport1(const char *, const char *, int, void *, void*, const char *);

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

#ifndef __VCS_IMPORT_DPI_STUB_add32_128bit
#define __VCS_IMPORT_DPI_STUB_add32_128bit
__attribute__((weak)) void add32_128bit(/* INPUT */unsigned long long A_1, /* INPUT */unsigned long long A_2, /* INPUT */unsigned long long A_3, /* INPUT */unsigned long long A_4, /* INPUT */int A_5, /* INPUT */int A_6, /* OUTPUT */unsigned long long *A_7, /* OUTPUT */unsigned long long *A_8, /* OUTPUT */unsigned long long *A_9, /* OUTPUT */unsigned long long *A_10
)
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static void (*_vcs_dpi_fp_)(/* INPUT */unsigned long long A_1, /* INPUT */unsigned long long A_2, /* INPUT */unsigned long long A_3, /* INPUT */unsigned long long A_4, /* INPUT */int A_5, /* INPUT */int A_6, /* OUTPUT */unsigned long long *A_7, /* OUTPUT */unsigned long long *A_8, /* OUTPUT */unsigned long long *A_9, /* OUTPUT */unsigned long long *A_10
) = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (void (*)(unsigned long long A_1, unsigned long long A_2, unsigned long long A_3, unsigned long long A_4, int A_5, int A_6, unsigned long long* A_7, unsigned long long* A_8, unsigned long long* A_9, unsigned long long* A_10)) dlsym(RTLD_NEXT, "add32_128bit");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        _vcs_dpi_fp_(A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10);
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "add32_128bit");
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_add32_128bit */

#ifndef __VCS_IMPORT_DPI_STUB_add8_128bit
#define __VCS_IMPORT_DPI_STUB_add8_128bit
__attribute__((weak)) void add8_128bit(/* INPUT */unsigned long long A_1, /* INPUT */unsigned long long A_2, /* INPUT */unsigned long long A_3, /* INPUT */unsigned long long A_4, /* INPUT */unsigned long long A_5, /* INPUT */unsigned long long A_6, /* INPUT */int A_7, /* INPUT */int A_8, /* INPUT */int A_9, /* OUTPUT */unsigned long long *A_10, 
/* OUTPUT */unsigned long long *A_11, /* OUTPUT */unsigned long long *A_12, /* OUTPUT */unsigned long long *A_13, /* OUTPUT */unsigned long long *A_14, /* OUTPUT */unsigned long long *A_15)
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static void (*_vcs_dpi_fp_)(/* INPUT */unsigned long long A_1, /* INPUT */unsigned long long A_2, /* INPUT */unsigned long long A_3, /* INPUT */unsigned long long A_4, /* INPUT */unsigned long long A_5, /* INPUT */unsigned long long A_6, /* INPUT */int A_7, /* INPUT */int A_8, /* INPUT */int A_9, /* OUTPUT */unsigned long long *A_10, 
/* OUTPUT */unsigned long long *A_11, /* OUTPUT */unsigned long long *A_12, /* OUTPUT */unsigned long long *A_13, /* OUTPUT */unsigned long long *A_14, /* OUTPUT */unsigned long long *A_15) = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (void (*)(unsigned long long A_1, unsigned long long A_2, unsigned long long A_3, unsigned long long A_4, unsigned long long A_5, unsigned long long A_6, int A_7, int A_8, int A_9, unsigned long long* A_10, unsigned long long* A_11, unsigned long long* A_12, unsigned long long* A_13, unsigned long long* A_14, unsigned long long* A_15)) dlsym(RTLD_NEXT, "add8_128bit");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        _vcs_dpi_fp_(A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12, A_13, A_14, A_15);
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "add8_128bit");
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_add8_128bit */


#ifdef __cplusplus
}
#endif

