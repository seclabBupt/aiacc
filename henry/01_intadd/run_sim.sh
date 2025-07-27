#!/bin/bash

# INTADD 仿真脚本（修改版）
# 支持 VCS 编译、仿真和覆盖率分析

# 脚本路径变量
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PROJ_ROOT=$SCRIPT_DIR
TB_FILE=$PROJ_ROOT/intadd_tb.v
DUT_FILE=$PROJ_ROOT/intadd.v
ADD8_FILE=$PROJ_ROOT/add8.v
ADD32_FILE=$PROJ_ROOT/add32.v
DPI_C_FILE=$PROJ_ROOT/softfloat_dpi.c

# 编译输出目录
OUTPUT_DIR=$PROJ_ROOT/sim_output

# 清理并创建输出目录
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# DPI 共享库名称
DPI_SO_NAME=libintadd_softfloat.so

# 清理旧的输出文件
rm -f $DPI_SO_NAME
rm -rf *.daidir
rm -rf *.vdb
rm -f .vcs.timestamp
rm -f simv
rm -f ucli.key

echo "=========================================="
echo "         INTADD 仿真脚本启动"
echo "=========================================="

# 检查必要文件是否存在
if [ ! -f "$DUT_FILE" ]; then
    echo "错误: 找不到 DUT 文件 $DUT_FILE"
    exit 1
fi

if [ ! -f "$TB_FILE" ]; then
    echo "错误: 找不到测试台文件 $TB_FILE"
    exit 1
fi

if [ ! -f "$DPI_C_FILE" ]; then
    echo "警告: 找不到 DPI-C 文件 $DPI_C_FILE，将跳过 DPI 编译"
    USE_DPI=false
else
    USE_DPI=true
fi

# 步骤 1: 编译 DPI-C 文件（如果存在）
if [ "$USE_DPI" = true ]; then
    echo "步骤 1: 编译 DPI-C 共享库..."
    /home/zwz/henry/SMC/compile_softfloat_dpi.sh $DPI_C_FILE $DPI_SO_NAME

    if [ $? -ne 0 ]; then
        echo "错误: DPI-C 文件编译失败"
        exit 1
    fi

    echo "成功创建共享库: $PWD/$DPI_SO_NAME"
else
    echo "步骤 1: 跳过 DPI-C 编译（文件不存在）"
fi

# 获取 SoftFloat 包含路径
SOFTFLOAT_INCLUDE="/home/zwz/henry/SMC/berkeley-softfloat-3-master/source/include"

# 步骤 2: 编译 Verilog 文件
echo "步骤 2: 使用 Synopsys VCS 编译 Verilog 文件..."

# 构建 VCS 编译命令
VCS_CMD="vcs -sverilog +v2k -full64 +fsdb -timescale=1ns/1ps"
VCS_CMD="$VCS_CMD -cm line+cond+fsm+branch+tgl"
VCS_CMD="$VCS_CMD -debug_access+all"
VCS_CMD="$VCS_CMD +vpi"
VCS_CMD="$VCS_CMD -Mupdate"

# 添加文件
VCS_CMD="$VCS_CMD $DUT_FILE $TB_FILE"

# 如果子模块文件存在，则加到编译命令里
if [ -f "$ADD8_FILE" ]; then
    echo "找到 add8.v，添加到编译文件列表"
    VCS_CMD="$VCS_CMD $ADD8_FILE"
else
    echo "提示: 未找到 add8.v，跳过"
fi

if [ -f "$ADD32_FILE" ]; then
    echo "找到 add32.v，添加到编译文件列表"
    VCS_CMD="$VCS_CMD $ADD32_FILE"
else
    echo "提示: 未找到 add32.v，跳过"
fi

# 如果有 DPI，添加编译参数
if [ "$USE_DPI" = true ]; then
    VCS_CMD="$VCS_CMD -CFLAGS \"-I$SOFTFLOAT_INCLUDE\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-Wl,-rpath,$(pwd)\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-L$(pwd)\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-lintadd_softfloat\""
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
echo "开始仿真，预计时间: 30-60秒..."

./simv -l sim.log -cm line+cond+fsm+branch+tgl +fsdb+autoflush

if [ $? -ne 0 ]; then
    echo "错误: VCS 仿真失败，查看 $OUTPUT_DIR/sim.log 获取详情"
    if [ -f "sim.log" ]; then
        echo "仿真日志最后10行:"
        tail -10 sim.log
    fi
    exit 1
else
    echo "仿真成功完成"
    echo "仿真日志: $OUTPUT_DIR/sim.log"
fi

# 波形检查
echo "步骤 4: 检查波形文件..."
if [ -f "intadd_tb.fsdb" ]; then
    FSDB_SIZE=$(du -h intadd_tb.fsdb | cut -f1)
    echo "FSDB波形文件生成成功: intadd_tb.fsdb (大小: $FSDB_SIZE)"
    echo "查看波形命令示例:"
    echo "  verdi -fsdb $OUTPUT_DIR/intadd_tb.fsdb &"
else
    echo "提示: 未找到 FSDB 波形文件"
fi

echo ""
echo "=========================================="
echo "         仿真脚本执行完成"
echo "=========================================="
echo "所有仿真结果都在: $OUTPUT_DIR/"
echo ""

# 回到根目录
cd $PROJ_ROOT
