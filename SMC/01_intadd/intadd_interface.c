#include <stdint.h>
#include <stdio.h>

void add32_128bit(
    unsigned long long src0_high, unsigned long long src0_low,
    unsigned long long src1_high, unsigned long long src1_low,
    int sign_s0, int sign_s1,
    unsigned long long* dst_high, unsigned long long* dst_low
) {
    uint32_t a[4], b[4], res[4];
    int i;
    // 拆分输入
    a[0] = (uint32_t)(src0_low      );
    a[1] = (uint32_t)(src0_low >> 32);
    a[2] = (uint32_t)(src0_high     );
    a[3] = (uint32_t)(src0_high >> 32);
    b[0] = (uint32_t)(src1_low      );
    b[1] = (uint32_t)(src1_low >> 32);
    b[2] = (uint32_t)(src1_high     );
    b[3] = (uint32_t)(src1_high >> 32);

    printf("C add32_128bit input:\n");
    printf("  src0_high=0x%016llx src0_low=0x%016llx\n", src0_high, src0_low);
    printf("  src1_high=0x%016llx src1_low=0x%016llx\n", src1_high, src1_low);
    printf("  sign_s0=%d sign_s1=%d\n", sign_s0, sign_s1);

    for (i = 0; i < 4; ++i) {
        int64_t s0 = sign_s0 ? (int32_t)a[i] : (uint64_t)a[i];
        int64_t s1 = sign_s1 ? (int32_t)b[i] : (uint64_t)b[i];
        int64_t sum = s0 + s1;
        if (sign_s0 || sign_s1) {
            // 溢出检测：同号加，结果符号变
            if ((s0 > 0 && s1 > 0 && sum < 0) || (s0 < 0 && s1 < 0 && sum >= 0)) {
                sum = 0x7FFFFFFFLL;
            } else {
                sum &= 0xFFFFFFFFLL;
            }
        } else {
            sum &= 0xFFFFFFFFLL;
        }
        res[i] = (uint32_t)sum;
    }
    // 拼回128位
    *dst_low  = ((uint64_t)res[1] << 32) | res[0];
    *dst_high = ((uint64_t)res[3] << 32) | res[2];
    printf("  dst_low=0x%016llx dst_high=0x%016llx\n", *dst_high, *dst_low);
}

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

