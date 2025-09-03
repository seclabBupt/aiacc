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

#ifndef __VCS_IMPORT_DPI_STUB_dpi_f16_mul
#define __VCS_IMPORT_DPI_STUB_dpi_f16_mul
__attribute__((weak)) unsigned short int dpi_f16_mul(/* INPUT */unsigned short int A_1, /* INPUT */unsigned short int A_2)
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned short int (*_vcs_dpi_fp_)(/* INPUT */unsigned short int A_1, /* INPUT */unsigned short int A_2) = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned short int (*)(unsigned short int A_1, unsigned short int A_2)) dlsym(RTLD_NEXT, "dpi_f16_mul");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_(A_1, A_2);
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_f16_mul");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_f16_mul */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_f32_mul
#define __VCS_IMPORT_DPI_STUB_dpi_f32_mul
__attribute__((weak)) unsigned int dpi_f32_mul(/* INPUT */unsigned int A_1, /* INPUT */unsigned int A_2)
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)(/* INPUT */unsigned int A_1, /* INPUT */unsigned int A_2) = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)(unsigned int A_1, unsigned int A_2)) dlsym(RTLD_NEXT, "dpi_f32_mul");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_(A_1, A_2);
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_f32_mul");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_f32_mul */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_inexact_flag
#define __VCS_IMPORT_DPI_STUB_dpi_get_inexact_flag
__attribute__((weak)) unsigned int dpi_get_inexact_flag()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_inexact_flag");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_inexact_flag");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_inexact_flag */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_underflow_flag
#define __VCS_IMPORT_DPI_STUB_dpi_get_underflow_flag
__attribute__((weak)) unsigned int dpi_get_underflow_flag()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_underflow_flag");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_underflow_flag");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_underflow_flag */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_overflow_flag
#define __VCS_IMPORT_DPI_STUB_dpi_get_overflow_flag
__attribute__((weak)) unsigned int dpi_get_overflow_flag()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_overflow_flag");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_overflow_flag");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_overflow_flag */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_infinite_flag
#define __VCS_IMPORT_DPI_STUB_dpi_get_infinite_flag
__attribute__((weak)) unsigned int dpi_get_infinite_flag()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_infinite_flag");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_infinite_flag");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_infinite_flag */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_invalid_flag
#define __VCS_IMPORT_DPI_STUB_dpi_get_invalid_flag
__attribute__((weak)) unsigned int dpi_get_invalid_flag()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_invalid_flag");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_invalid_flag");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_invalid_flag */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_get_exception_flags
#define __VCS_IMPORT_DPI_STUB_dpi_get_exception_flags
__attribute__((weak)) unsigned int dpi_get_exception_flags()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static unsigned int (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (unsigned int (*)()) dlsym(RTLD_NEXT, "dpi_get_exception_flags");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        return _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_get_exception_flags");
        return 0;
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_get_exception_flags */

#ifndef __VCS_IMPORT_DPI_STUB_dpi_clear_exception_flags
#define __VCS_IMPORT_DPI_STUB_dpi_clear_exception_flags
__attribute__((weak)) void dpi_clear_exception_flags()
{
    static int _vcs_dpi_stub_initialized_ = 0;
    static void (*_vcs_dpi_fp_)() = NULL;
    if (!_vcs_dpi_stub_initialized_) {
        _vcs_dpi_fp_ = (void (*)()) dlsym(RTLD_NEXT, "dpi_clear_exception_flags");
        _vcs_dpi_stub_initialized_ = 1;
    }
    if (_vcs_dpi_fp_) {
        _vcs_dpi_fp_();
    } else {
        const char *fileName;
        int lineNumber;
        svGetCallerInfo(&fileName, &lineNumber);
        vcsMsgReport1("DPI-DIFNF", fileName, lineNumber, 0, 0, "dpi_clear_exception_flags");
    }
}
#endif /* __VCS_IMPORT_DPI_STUB_dpi_clear_exception_flags */


#ifdef __cplusplus
}
#endif

