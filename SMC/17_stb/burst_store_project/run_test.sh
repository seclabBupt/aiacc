#!/bin/bash

# FPADD 仿真脚本
# 支持 VCS 编译、仿真和覆盖率分析

# 路径变量
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PROJ_ROOT=$SCRIPT_DIR
SRC_DIR=$PROJ_ROOT/src
TB_DIR=$PROJ_ROOT/tb
DPI_C_FILE=$PROJ_ROOT/softfloat_dpi.c

# DUT和TB文件
DUT_FILE=$SRC_DIR/float_adder.v
DUT_FILE32=$SRC_DIR/float32_adder.v
DUT_FILE16=$SRC_DIR/float16_adder.v
TB_FILE=$TB_DIR/float_adder_tb.v

# 输出目录
OUTPUT_DIR=$PROJ_ROOT/sim_output

# 清理并创建输出目录
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# DPI 共享库名称
DPI_SO_NAME=libfpadd_softfloat.so

echo "=========================================="
echo "         FPADD 仿真脚本启动"
echo "=========================================="

# 检查必要文件
for f in "$DUT_FILE" "$DUT_FILE32" "$DUT_FILE16" "$TB_FILE"; do
    if [ ! -f "$f" ]; then
        echo "错误: 找不到文件 $f"
        exit 1
    fi
done

if [ ! -f "$DPI_C_FILE" ]; then
    echo "警告: 找不到DPI-C文件 $DPI_C_FILE,将跳过DPI编译"
    USE_DPI=false
else
    USE_DPI=true
fi

# 步骤 1: 编译 DPI-C 文件（如果存在）
if [ "$USE_DPI" = true ]; then
    echo "步骤 1: 编译 DPI-C 共享库..."
    echo "使用一体化脚本编译 DPI-C 文件..."
    $PROJ_ROOT/compile_softfloat_dpi.sh $DPI_C_FILE $DPI_SO_NAME

    if [ $? -ne 0 ]; then
        echo "错误: DPI-C 文件编译失败"
        exit 1
    fi

    echo "成功创建共享库: $PWD/$DPI_SO_NAME"
else
    echo "步骤 1: 跳过 DPI-C 编译（文件不存在）"
fi

# 获取 SoftFloat 包含路径（自动检测）
SOFTFLOAT_INCLUDE="$PROJ_ROOT/../berkeley-softfloat-3-master/source/include"
if [ ! -d "$SOFTFLOAT_INCLUDE" ]; then
    SOFTFLOAT_INCLUDE="$PROJ_ROOT/../../berkeley-softfloat-3-master/source/include"
fi
if [ ! -d "$SOFTFLOAT_INCLUDE" ]; then
    SOFTFLOAT_INCLUDE="$PROJ_ROOT/berkeley-softfloat-3-master/source/include"
fi

# 步骤 2: 编译 Verilog 文件
echo "步骤 2: 使用 Synopsys VCS 编译 Verilog 文件..."

VCS_CMD="vcs -sverilog +v2k -full64 +fsdb -timescale=1ns/1ps"
VCS_CMD="$VCS_CMD -cm line+cond+fsm+branch+tgl"
VCS_CMD="$VCS_CMD -debug_access+all"
VCS_CMD="$VCS_CMD +vpi"
VCS_CMD="$VCS_CMD -Mupdate"
VCS_CMD="$VCS_CMD $DUT_FILE $DUT_FILE32 $DUT_FILE16 $TB_FILE"

if [ "$USE_DPI" = true ]; then
    VCS_CMD="$VCS_CMD -CFLAGS \"-I$SOFTFLOAT_INCLUDE\""
    VCS_CMD="$VCS_CMD -sv_lib $PWD/$DPI_SO_NAME"
fi

VCS_CMD="$VCS_CMD -o simv"

echo "VCS 编译命令: $VCS_CMD"
eval $VCS_CMD

if [ $? -ne 0 ]; then
    echo "错误: VCS Verilog 编译失败"
    exit 1
else
    echo "VCS 编译成功完成"
fi

# 步骤 3: 运行仿真
echo "步骤 3: 运行仿真..."

./simv -l sim.log -cm line+cond+fsm+branch+tgl +fsdb+autoflush

if [ $? -ne 0 ]; then
    echo "错误: VCS 仿真失败，查看 $OUTPUT_DIR/sim.log 获取详情"
    exit 1
else
    echo "仿真成功完成"
    echo "仿真日志: $OUTPUT_DIR/sim.log"
fi

# 步骤 4: 生成和查看覆盖率报告
echo "步骤 4: 生成覆盖率报告..."

if [ -d "simv.vdb" ]; then
    urg -dir simv.vdb -format both -report coverage_report
    if [ $? -ne 0 ]; then
        echo "警告: 覆盖率报告生成失败"
    else
        echo "覆盖率报告生成完成"
        echo "报告位置: $OUTPUT_DIR/coverage_report/"
        echo "HTML报告: $OUTPUT_DIR/coverage_report/dashboard.html"
        echo "文本报告: $OUTPUT_DIR/coverage_report/dashboard.txt"
    fi
else
    echo "警告: 找不到覆盖率数据库 simv.vdb"
fi

# 步骤 5: 波形文件检查
echo "步骤 5: 检查波形文件..."
if [ -f "tb_fpadd.fsdb" ]; then
    FSDB_SIZE=$(du -h tb_fpadd.fsdb | cut -f1)
    echo "FSDB波形文件生成成功: tb_fpadd.fsdb (大小: $FSDB_SIZE)"
else
    echo "警告: 未找到FSDB波形文件"
fi

# 回到项目根目录
cd $PROJ_ROOT

echo ""
echo "=========================================="
echo "         仿真脚本执行完成"
echo "=========================================="
echo "所有仿真结果都在: $OUTPUT_DIR/"
echo ""
echo "主要文件列表:"
echo "  - 仿真可执行文件: $OUTPUT_DIR/simv"
echo "  - 仿真日志: $OUTPUT_DIR/sim.log"
echo "  - 测试结果: $OUTPUT_DIR/results.txt"
echo "  - FSDB波形: $OUTPUT_DIR/tb_fpadd.fsdb"
echo "  - 覆盖率报告: $OUTPUT_DIR/coverage_report/"
echo ""

exit 0