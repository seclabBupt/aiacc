#!/bin/bash

echo "清理之前的仿真文件..."
rm -rf sim_output
mkdir -p sim_output
cd sim_output

# 使用一体化脚本编译 DPI-C 文件
/home/Sunny/SMC/compile_softfloat_dpi.sh ../csrc/softfloat_fp32_dpi.c libruntime.so

if [ $? -ne 0 ]; then
    echo "错误: DPI-C 文件编译失败"
    exit 1
fi

# 复制库文件到上级目录，确保VCS可以找到它
#cp ./libruntime.so ../libruntime.so

echo "设置库路径..."
# 设置LD_LIBRARY_PATH以便VCS能找到共享库
LIB_PATH=$(pwd)
export LD_LIBRARY_PATH="$LIB_PATH:$PWD/..:$LD_LIBRARY_PATH"

# 获取 SoftFloat 包含路径（从一体化脚本的输出中获取）
SOFTFLOAT_INCLUDE="/home/Sunny/SMC/berkeley-softfloat-3-master/source/include"

echo "运行 VCS 编译..."
# 使用 VCS 编译 Verilog 文件
vcs -sverilog +v2k -full64 +fsdb -debug_access+all \
    -kdb \
    -timescale=1ns/1ps \
    +incdir+.. \
    +define+DUMP_FSDB \
    ../vsrc/tb_fp32_adder_tree_8_inputs_softfloat.v \
    ../vsrc/fp32_adder_tree_8_inputs.v \
    ../vsrc/fp32_unpacker.v \
    ../vsrc/fp32_aligner.v \
    ../vsrc/wallace_tree_8_inputs.v \
    ../vsrc/full_adder.v \
    ../vsrc/final_adder.v \
    ../vsrc/fp32_normalizer_rounder.v \
    ../vsrc/fp32_packer.v \
    -o simv_softfloat \
    -CFLAGS "-I$SOFTFLOAT_INCLUDE" \
    $LIB_PATH/libruntime.so

if [ $? -ne 0 ]; then
    echo "错误: VCS 编译失败"
    exit 1
fi

echo "运行仿真..."
# 确保库路径正确设置
export LD_LIBRARY_PATH=".:$LD_LIBRARY_PATH"

./simv_softfloat

if [ $? -ne 0 ]; then
    echo "错误: 仿真运行失败"
    exit 1
fi

echo "仿真完成！查看 sim_softfloat.log 文件获取结果。"

















