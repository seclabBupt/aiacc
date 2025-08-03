// SoftFloat DPI-C 接口文件，用于 FPMUL 模块验证
// 提供 IEEE 754 标准的浮点乘法参考实现

#include <stdio.h>
#include <stdint.h>
#include "softfloat.h"

// SoftFloat 使用全局变量来管理舍入模式和异常标志
// 这些变量在 softfloat.h 中声明为 extern

// 设置SoftFloat的舍入模式
void set_softfloat_rounding_mode(uint32_t mode) {
    softfloat_roundingMode = mode;
}

// 清除SoftFloat的异常标志
void clear_softfloat_flags() {
    softfloat_exceptionFlags = 0;
}

// 获取SoftFloat的异常标志
uint32_t get_softfloat_flags() {
    return softfloat_exceptionFlags;
}

/***
#define softfloat_flag_inexact   1   // 不精确
#define softfloat_flag_underflow 2   // 下溢
#define softfloat_flag_overflow  4   // 上溢
#define softfloat_flag_infinite  8   // 无穷大
#define softfloat_flag_invalid   16  // 无效
***/

// DPI-C 函数实现 - 避免命名冲突，使用不同的函数名

// FP16 乘法函数
uint16_t dpi_f16_mul(uint16_t a, uint16_t b) {
    // 清除异常标志
    softfloat_exceptionFlags = 0;
    
    // 转换为 SoftFloat 格式
    float16_t fa, fb, result;
    fa.v = a;
    fb.v = b;
    
    // 执行乘法
    result = f16_mul(fa, fb);
    
    return result.v;
}

// FP32 乘法函数  
uint32_t dpi_f32_mul(uint32_t a, uint32_t b) {
    // 清除异常标志
    softfloat_exceptionFlags = 0;
    
    // 转换为 SoftFloat 格式
    float32_t fa, fb, result;
    fa.v = a;
    fb.v = b;
    
    // 执行乘法
    result = f32_mul(fa, fb);
    
    return result.v;
}

// 设置异常标志
void dpi_softfloat_raiseFlags(uint32_t flags) {
    softfloat_exceptionFlags |= (uint_fast8_t)flags;
}

// 获取异常标志常量
uint32_t dpi_get_inexact_flag() {
    return softfloat_flag_inexact;
}

uint32_t dpi_get_underflow_flag() {
    return softfloat_flag_underflow;
}

uint32_t dpi_get_overflow_flag() {
    return softfloat_flag_overflow;
}

uint32_t dpi_get_infinite_flag() {
    return softfloat_flag_infinite;
}

uint32_t dpi_get_invalid_flag() {
    return softfloat_flag_invalid;
}

// 获取当前异常标志
uint32_t dpi_get_exception_flags() {
    return (uint32_t)softfloat_exceptionFlags;
}

// 清除异常标志
void dpi_clear_exception_flags() {
    softfloat_exceptionFlags = 0;
}

// 设置舍入模式
void dpi_set_rounding_mode(uint32_t mode) {
    softfloat_roundingMode = (uint_fast8_t)mode;
}

// 获取舍入模式
uint32_t dpi_get_rounding_mode() {
    return (uint32_t)softfloat_roundingMode;
}
