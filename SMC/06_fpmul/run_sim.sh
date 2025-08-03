#!/bin/bash

# FPMUL 仿真脚本
# 支持 VCS 编译、仿真和覆盖率分析

# 脚本路径变量
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PROJ_ROOT=$SCRIPT_DIR
TB_FILE=$PROJ_ROOT/vsrc/tb_fpmul.v
DUT_FILE=$PROJ_ROOT/vsrc/fpmul.v
DPI_C_FILE=$PROJ_ROOT/csrc/softfloat_dpi.c

# 编译输出目录
OUTPUT_DIR=$PROJ_ROOT/sim_output

# 清理并创建输出目录
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# DPI 共享库名称
DPI_SO_NAME=libfpmul_softfloat.so

echo "=========================================="
echo "         FPMUL 仿真脚本启动"
echo "=========================================="

# 检查必要文件是否存在
if [ ! -f "$DUT_FILE" ]; then
    echo "错误: 找不到DUT文件 $DUT_FILE"
    exit 1
fi

if [ ! -f "$TB_FILE" ]; then
    echo "错误: 找不到测试台文件 $TB_FILE"
    exit 1
fi

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
    /home/Sunny/SMC/compile_softfloat_dpi.sh $DPI_C_FILE $DPI_SO_NAME

    if [ $? -ne 0 ]; then
        echo "错误: DPI-C 文件编译失败"
        exit 1
    fi

    echo "成功创建共享库: $PWD/$DPI_SO_NAME"
else
    echo "步骤 1: 跳过 DPI-C 编译（文件不存在）"
fi

# 获取 SoftFloat 包含路径
SOFTFLOAT_INCLUDE="/home/Sunny/SMC/berkeley-softfloat-3-master/source/include"

# 步骤 2: 编译 Verilog 文件
echo "步骤 2: 使用 Synopsys VCS 编译 Verilog 文件..."

# 构建VCS编译命令
VCS_CMD="vcs -sverilog +v2k -full64 +fsdb -timescale=1ns/1ps"
VCS_CMD="$VCS_CMD -cm line+cond+fsm+branch+tgl"
VCS_CMD="$VCS_CMD -debug_access+all"
VCS_CMD="$VCS_CMD +vpi"
VCS_CMD="$VCS_CMD -Mupdate"
VCS_CMD="$VCS_CMD $DUT_FILE $TB_FILE $PROJ_ROOT/vsrc/lzc.v"  

# 添加DPI相关选项
if [ "$USE_DPI" = true ]; then
    VCS_CMD="$VCS_CMD -CFLAGS \"-I$SOFTFLOAT_INCLUDE\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-Wl,-rpath,$(pwd)\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-L$(pwd)\""
    VCS_CMD="$VCS_CMD -LDFLAGS \"-lfpmul_softfloat\""
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
    # 生成HTML格式的覆盖率报告
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
if [ -f "tb_fpmul.fsdb" ]; then
    FSDB_SIZE=$(du -h tb_fpmul.fsdb | cut -f1)
    echo "FSDB波形文件生成成功: tb_fpmul.fsdb (大小: $FSDB_SIZE)"
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
echo "  - 测试日志: $OUTPUT_DIR/tb_fpmul.log"
if [ -f "$OUTPUT_DIR/tb_fpmul_errors.log" ]; then
echo "  - 错误日志: $OUTPUT_DIR/tb_fpmul_errors.log"
fi
echo "  - FSDB波形: $OUTPUT_DIR/tb_fpmul.fsdb"
echo "  - 覆盖率报告: $OUTPUT_DIR/coverage_report/"
echo ""

# 最终状态检查
if [ -f "$OUTPUT_DIR/tb_fpmul.log" ]; then
    FINAL_STATUS=$(grep -E "(所有测试都通过|个测试失败)" "$OUTPUT_DIR/tb_fpmul.log" | tail -1)
    if echo "$FINAL_STATUS" | grep -q "所有测试都通过"; then
        echo "🎉 最终结果: 所有测试通过!"
        exit 0
    elif echo "$FINAL_STATUS" | grep -q "失败"; then
        echo "❌ 最终结果: 部分测试失败，请检查错误日志"
        exit 1
    else
        echo "⚠️  最终结果: 无法确定测试状态"
        exit 1
    fi
else
    echo "⚠️  警告: 未找到测试结果文件"
    exit 1
fi
