#include <stdint.h>
#include <stdio.h>

void add32_128bit(
    unsigned long long src0_high, unsigned long long src0_low,
    unsigned long long src1_high, unsigned long long src1_low,
    int sign_s0, int sign_s1,
    unsigned long long* dst_high, unsigned long long* dst_low,
    unsigned long long* st_high, unsigned long long* st_low
) {
    uint32_t a[4], b[4], res[4];
    int i;
    // 拆分输入
    a[0] = (uint32_t)(src0_low);
    a[1] = (uint32_t)(src0_low >> 32);
    a[2] = (uint32_t)(src0_high);
    a[3] = (uint32_t)(src0_high >> 32);
    b[0] = (uint32_t)(src1_low);
    b[1] = (uint32_t)(src1_low >> 32);
    b[2] = (uint32_t)(src1_high);
    b[3] = (uint32_t)(src1_high >> 32);

    // 初始化状态寄存器
    *st_high = 0;
    *st_low = 0;

    for (i = 0; i < 4; ++i) {
        // 独立符号扩展
        int64_t s0 = sign_s0 ? (int32_t)a[i] : (uint64_t)a[i];
        int64_t s1 = sign_s1 ? (int32_t)b[i] : (uint64_t)b[i];
        int64_t sum = s0 + s1;
        
        // 处理溢出
        if (sign_s0 || sign_s1) {
            if ((s0 > 0 && s1 > 0 && sum < 0) || 
                (s0 < 0 && s1 < 0 && sum >= 0)) {
                res[i] = 0x7FFFFFFF; // 饱和处理
            } else {
                res[i] = (uint32_t)sum;
            }
        } else {
            res[i] = (uint32_t)sum;
        }
        
        // 精确的32位比较逻辑
        int gt = 0, eq = 0, ls = 0;
        if (sign_s0 || sign_s1) {
            // 有符号比较使用32位直接比较
            int32_t signed_a = (int32_t)a[i];
            int32_t signed_b = (int32_t)b[i];
            if (signed_a > signed_b) gt = 1;
            else if (signed_a < signed_b) ls = 1;
            else eq = 1;
        } else {
            // 无符号比较
            if (a[i] > b[i]) gt = 1;
            else if (a[i] < b[i]) ls = 1;
            else eq = 1;
        }
        
        // 状态位设置 - 严格按组放置
        uint64_t status_bits = 0;
        status_bits |= ((uint64_t)gt << 2);
        status_bits |= ((uint64_t)eq << 1);
        status_bits |= ((uint64_t)ls << 0);
        
        // 根据组号放置状态位
        switch(i) {
            case 0: *st_low  |= (status_bits << 0);  break;  // 位2:0
            case 1: *st_low  |= (status_bits << 32); break;  // 位34:32
            case 2: *st_high |= (status_bits << 0);  break;  // 位66:64
            case 3: *st_high |= (status_bits << 32); break;  // 位98:96
        }
    }
    
    // 拼回128位结果
    *dst_low  = ((uint64_t)res[1] << 32) | res[0];
    *dst_high = ((uint64_t)res[3] << 32) | res[2];
}

// add8_128bit 适配当前的add8.v实现
void add8_128bit(
    unsigned long long src0_high, unsigned long long src0_low,
    unsigned long long src1_high, unsigned long long src1_low,
    unsigned long long src2_high, unsigned long long src2_low,
    int sign_s0, int sign_s1, int sign_s2,
    unsigned long long* dst0_high, unsigned long long* dst0_low,
    unsigned long long* dst1_high, unsigned long long* dst1_low
) {
    uint8_t a[32], b[32], c[32];
    uint8_t res0[32], res1[32];
    int i;

    // 拆分为32个4bit段
    for (i = 0; i < 16; ++i) {
        a[i]      = (src0_low  >> (i*4)) & 0xF;
        b[i]      = (src1_low  >> (i*4)) & 0xF;
        c[i]      = (src2_low  >> (i*4)) & 0xF;
        a[i+16]   = (src0_high >> (i*4)) & 0xF;
        b[i+16]   = (src1_high >> (i*4)) & 0xF;
        c[i+16]   = (src2_high >> (i*4)) & 0xF;
    }

    for (i = 0; i < 32; ++i) {
        // 4bit符号扩展到8bit
        int8_t s0 = sign_s0 ? ((a[i] & 0x8) ? (a[i] | 0xF0) : a[i]) : a[i];
        int8_t s1 = sign_s1 ? ((b[i] & 0x8) ? (b[i] | 0xF0) : b[i]) : b[i];
        int8_t s2 = sign_s2 ? ((c[i] & 0x8) ? (c[i] | 0xF0) : c[i]) : c[i];
        int16_t sum = (int16_t)s0 + (int16_t)s1 + (int16_t)s2;

        // 饱和到8bit
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;

        uint8_t sum8 = (uint8_t)sum;
        res0[i] = sum8 & 0xF;         // 低4位
        res1[i] = (sum8 >> 4) & 0xF;  // 高4位
    }

    // 拼回128位
    *dst0_low = 0; *dst0_high = 0;
    *dst1_low = 0; *dst1_high = 0;
    for (i = 0; i < 16; ++i) {
        *dst0_low  |= ((uint64_t)res0[i]      & 0xF) << (i*4);
        *dst0_high |= ((uint64_t)res0[i+16]   & 0xF) << (i*4);
        *dst1_low  |= ((uint64_t)res1[i]      & 0xF) << (i*4);
        *dst1_high |= ((uint64_t)res1[i+16]   & 0xF) << (i*4);
    }
}

