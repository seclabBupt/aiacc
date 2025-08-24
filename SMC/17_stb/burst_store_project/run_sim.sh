#!/bin/bash

# Burst Store 仿真脚本（无DPI/SoftFloat依赖）

# 路径变量
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PROJ_ROOT=$SCRIPT_DIR
RTL_DIR=$PROJ_ROOT/rtl
TB_DIR=$PROJ_ROOT/tb

# DUT 和 TB 文件
DUT_FILE=$RTL_DIR/burst_store.v
TB_FILE=$TB_DIR/burst_store_tb.v
MEM_MODEL_FILE=$TB_DIR/mem_model.v
AXI_HOST_FILE=$RTL_DIR/axi_stb.v          # 主机协议
AXI_SLAVE_FILE=$RTL_DIR/axi_stb_s.v       # 从机协议

# 输出目录
OUTPUT_DIR=$PROJ_ROOT/sim_output

# 清理并创建输出目录
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "         Burst Store 仿真脚本启动"
echo "=========================================="

# 检查必要文件
for f in "$DUT_FILE" "$MEM_MODEL_FILE" "$TB_FILE" "$AXI_HOST_FILE" "$AXI_SLAVE_FILE"; do
    if [ ! -f "$f" ]; then
        echo "错误: 找不到文件 $f"
        exit 1
    fi
done

# 步骤 1: 编译 Verilog 文件
echo "步骤 1: 使用 Synopsys VCS 编译 Verilog 文件..."
echo "编译文件列表:"
echo "  - $MEM_MODEL_FILE"
echo "  - $AXI_SLAVE_FILE"
echo "  - $AXI_HOST_FILE"
echo "  - $DUT_FILE"
echo "  - $TB_FILE"

vcs -sverilog +v2k -full64 -timescale=1ns/1ps -debug_access+all \
    -cm line+cond+tgl+fsm+branch \
    "$MEM_MODEL_FILE" "$AXI_SLAVE_FILE" "$AXI_HOST_FILE" "$DUT_FILE" "$TB_FILE" -o simv

if [ $? -ne 0 ]; then
    echo "错误: VCS Verilog 编译失败"
    exit 1
else
    echo "VCS 编译成功完成"
fi

# 步骤 2: 运行仿真
echo "步骤 2: 运行仿真..."

./simv -cm line+cond+tgl+fsm+branch -l sim.log

if [ $? -ne 0 ]; then
    echo "错误: VCS 仿真失败，查看 $OUTPUT_DIR/sim.log 获取详情"
    exit 1
else
    echo "仿真成功完成"
    echo "仿真日志: $OUTPUT_DIR/sim.log"
fi

# 步骤 3: 检查波形文件
echo "步骤 3: 检查波形文件..."
if [ -f "burst_store.vcd" ]; then
    VCD_SIZE=$(du -h burst_store.vcd | cut -f1)
    echo "VCD波形文件生成成功: burst_store.vcd (大小: $VCD_SIZE)"
else
    echo "警告: 未找到VCD波形文件"
fi

# 步骤 4: 检查数据文件
echo "步骤 4: 检查数据文件..."
for data_file in "src_data.txt" "mem_data.txt"; do
    if [ -f "$data_file" ]; then
        DATA_SIZE=$(du -h "$data_file" | cut -f1)
        echo "数据文件生成成功: $data_file (大小: $DATA_SIZE)"
    else
        echo "警告: 未找到数据文件 $data_file"
    fi
done

# 可选：自动打开GTKWave
if [ -f "$PROJ_ROOT/waves.gtkw" ] && [ -f "burst_store.vcd" ]; then
    echo "自动启动GTKWave查看波形..."
    gtkwave burst_store.vcd "$PROJ_ROOT/waves.gtkw" &
fi

# 步骤 5: 覆盖率报告生成
echo "步骤 5: 生成覆盖率报告..."
if [ -d "simv.vdb" ]; then
    urg -dir simv.vdb -format both -report coverage_report
    if [ $? -ne 0 ]; then
        echo "警告: 覆盖率报告生成失败"
    else
        echo "覆盖率报告生成完成"
        echo "HTML报告: $OUTPUT_DIR/coverage_report/dashboard.html"
        echo "文本报告: $OUTPUT_DIR/coverage_report/dashboard.txt"
    fi
else
    echo "警告: 找不到覆盖率数据库 simv.vdb"
fi

# 回到项目根目录
cd "$PROJ_ROOT"

echo ""
echo "=========================================="
echo "         仿真脚本执行完成"
echo "=========================================="
echo "所有仿真结果都在: $OUTPUT_DIR/"
echo ""
echo "主要文件列表:"
echo "  - 仿真可执行文件: $OUTPUT_DIR/simv"
echo "  - 仿真日志: $OUTPUT_DIR/sim.log"
echo "  - VCD波形: $OUTPUT_DIR/burst_store.vcd"
echo "  - 源数据: $OUTPUT_DIR/src_data.txt"
echo "  - 内存数据: $OUTPUT_DIR/mem_data.txt"
echo "  - 覆盖率报告: $OUTPUT_DIR/coverage_report/"
echo ""
 
exit 0