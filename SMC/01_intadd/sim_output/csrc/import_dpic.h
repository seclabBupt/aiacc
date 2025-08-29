
 extern void add32_128bit(/* INPUT */unsigned long long src0_high, /* INPUT */unsigned long long src0_low, /* INPUT */unsigned long long src1_high, /* INPUT */unsigned long long src1_low, /* INPUT */int sign_s0, /* INPUT */int sign_s1, /* OUTPUT */unsigned long long *dst_high, /* OUTPUT */unsigned long long *dst_low, /* OUTPUT */unsigned long long *st_high, /* OUTPUT */unsigned long long *st_low
);

 extern void add8_128bit(/* INPUT */unsigned long long src0_high, /* INPUT */unsigned long long src0_low, /* INPUT */unsigned long long src1_high, /* INPUT */unsigned long long src1_low, /* INPUT */unsigned long long src2_high, /* INPUT */unsigned long long src2_low, /* INPUT */int sign_s0, /* INPUT */int sign_s1, /* INPUT */int sign_s2, /* OUTPUT */unsigned long long *dst0_high, 
/* OUTPUT */unsigned long long *dst0_low, /* OUTPUT */unsigned long long *dst1_high, /* OUTPUT */unsigned long long *dst1_low, /* OUTPUT */unsigned long long *st_high, /* OUTPUT */unsigned long long *st_low);
