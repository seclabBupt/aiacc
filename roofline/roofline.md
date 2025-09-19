# Roofline整理
这里整理了zotero里关于roofline的文章。@Leah
## 1、Roofline Model
### 1.1、Roofline Model介绍
Roofline模型通过操作强度和脊点直观揭示系统瓶颈，指导优化策略选择。在多核架构多样化的背景下，该模型为程序员和架构师提供了统一的性能分析框架，尤其适用于科学计算、图形处理等内存密集型任务。未来可扩展至GPU、向量处理器等更多场景。
![roofline模型](https://haomenghit.github.io/2019/09/11/Roofline-Model%E4%BB%8B%E7%BB%8D/Figure1.png)
+ 横轴：操作强度（operational intensity）——每从 DRAM 搬运 1 Byte 数据到片内，程序能完成多少浮点运算（Operational Intensity=浮点操作数/DRAM访问字节数）。
+ 纵轴：可达浮点性能（GFlops/s），确定的芯片条件下能达到的浮点计算量的上限。内存每秒传输 B 字节，若每字节支持 I 次浮点操作，则最大性能为 I × B GFlops/s。（定值，由计算机决定）
+ 斜率：峰值内存带宽（GB/s），每秒能内存交换的上限。（定值，由计算机决定）
+ 水平线：硬件峰值浮点性能（计算能力上限）。
+ 斜线：内存带宽限制（内存能力上限，公式为 带宽 × 操作强度）。
+ 脊点（Ridge Point）：两线交点，反映系统的最小操作强度需求。
**当operational intensity<Ridge Point，内存瓶颈，减少DRAM访问或提升operational intensity**
**当operational intensity>Ridge Point，计算瓶颈，提升指令并行或平衡计算。**
### 1.2、Roofline Model论文里的应用
| 论文 | 论文内容 |
| :------: | :------: |
| Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures |  是Roofline模型的最早和最核心的介绍，它们奠定了Roofline分析方法的基础。其他论文则是在此基础上进行扩展、应用或改进。  |
| 8 Steps to 3.7 TFLOP/s on NVIDIA V100 GPU: Roofline Analysis and Other Tricks |  通过Roofline分析及其他优化技巧，详细介绍了如何在NVIDIA V100 GPU上将性能提升到3.7 TFLOP/s。  |
| Toward Automated Application Proﬁling on Cray Systems | 开发一种自动化工具，用于在Cray系统上对应用程序进行性能分析，以简化瓶颈识别过程。 |
| Performance Analysis and Benchmarking of a Temperature Downscaling Deep Learning Model |  本文对一个用于温度降尺度的深度学习模型的性能进行了分析和基准测试。 |
| Architectural Requirements for Deep Learning Workloads in HPC Environments | 本文探讨并定义了在高性能计算（HPC）环境中支持深度学习工作负载所需的体系结构要求。 |
| Preparing NERSC users for Cori, a Cray XC40 system with Intel many integrated cores | 本文为NERSC用户使用Cori系统（一个配备Intel MIC处理器的Cray XC40系统）提供了性能优化和使用指导。 |
| Studying performance portability of LAMMPS across diverse GPU‐based platforms | 本文研究了LAMMPS分子动力学模拟软件在不同GPU平台上的性能可移植性。 |
| A Case Study for Performance Portability Using OpenMP 4.5 | 本文通过一个案例研究，展示了如何使用OpenMP 4.5标准评估和实现代码的性能可移植性。 |
| Understanding Strong Scaling on GPUs Using Empirical Performance Saturation Size | 本文通过经验性性能饱和规模来理解GPU上应用程序的强扩展性行为。 |
| Applying the Roofline Performance Model to the Intel Xeon Phi Knights Landing Processor | 本文具体阐述了如何将Roofline性能模型应用于Intel Xeon Phi Knights Landing处理器进行性能分析。 |
| Performance Portability Evaluation of OpenCL Benchmarks across Intel and NVIDIA Platforms | 本文评估了OpenCL基准测试在Intel和NVIDIA平台上的性能可移植性。 |
| Accelerating Large-Scale GW Calculations on Hybrid GPU-CPU Systems | 本文研究了如何在混合GPU-CPU系统上加速大规模GW（格林函数-沃克方程）计算。 |
## 2、Cache-Aware Roofline Model
### 2.1、Cache-Aware Roofline Model介绍
该模型在经典Roofline的基础上考虑了缓存层次结构的影响，区分了不同层级缓存的带宽和延迟，从而更准确地反映实际性能。
[![Cache-Aware Roofline模型](https://s21.ax1x.com/2025/09/18/pVhJNy8.md.png)](https://imgse.com/i/pVhJNy8)
+ 核心思想：所有东西都从 CPU核心(Core)的角度 来看，计算 β (总传输字节数) 时，包含所有缓存层级之间以及缓存与核心之间传输的总字节数。无论数据最终是从L1、L2、L3还是DRAM来的，只要它被核心访问过，就算在内。
+ 操作强度 (I): I 定义为 总浮点操作数(φ) / 总传输字节数(β)。这个 I 是唯一的，适用于整个内存层次结构。
+ 带宽： 峰值带宽 B(β) 和峰值性能 F(φ) 不是固定值，它们会随着访问的数据量和计算量变化（比如，访问少量数据时可能达到L1的最高带宽，访问大量数据时受限于DRAM的带宽）。
+ 性能上限公式： Fa(I) = min{ B(β) * I, F(φ) }。这里 B(β) 和 F(φ) 是变化的函数。
+ 关键瓶颈带宽： 在内存瓶颈区域，性能主要受限于离核心最近的缓存（通常是L1缓存）到核心的带宽 (BL1-C)，因为这是数据到达计算单元的最后一步，也是最快的一步（如果数据在L1里）。
### 2.2、Cache-Aware Roofline Model扩展
#### 2.2.1、Energy-Efficiency Roofline Model
Energy-Efficiency Roofline 是 CARM 在能效维度的自然延伸，统一了性能、功耗与缓存层级的影响。其核心价值在于 量化微架构能效上限，并通过 AI 关联指导算法优化。
[![pVhJb6K.png](https://s21.ax1x.com/2025/09/18/pVhJb6K.png)](https://imgse.com/i/pVhJb6K)
+ 横轴（X轴）：算术强度（Arithmetic Intensity, AI）定义为 FLOPS/Byte，表示每字节数据流量可完成的浮点运算次数。
+ 纵轴（Y轴）：能效（Energy-Efficiency）单位为 FLOPS/Joule（每焦耳能量可完成的浮点运算次数），表示处理器执行计算的能量效率。
#### 2.2.2、Application-driven Cache-Aware Roofline Model
传统的屋顶模型，甚至是后来考虑了缓存的屋顶模型（Cache-Aware Roofline Model, CARM），其性能“屋顶”通常是基于硬件的理论峰值来设定的。这篇论文指出，这种方法可能过于理想化，因为真实世界的应用程序很少能达到理论峰值。原因在于，应用程序通常混合使用了多种不同类型的指令（如加法、乘法、FMA指令），并且数据访问模式也多种多样，这些因素都会影响实际可达到的性能。为此，提出了 adCARM (Application-driven Cache-Aware Roofline Model)。其核心思想是，性能模型的“屋顶”不应该是静态的、基于硬件理论峰值的，而应该是动态的、根据应用程序自身的特性来量身定制的，从而提供一个更精确、更现实的性能上界。
[![pVhJZz6.jpg](https://s21.ax1x.com/2025/09/18/pVhJZz6.jpg)](https://imgse.com/i/pVhJZz6)
+ 纵轴（Y轴）：性能（Performance），单位GFLOP/s（每秒十亿次浮点运算），表示应用程序在该算术强度下所能达到的最大浮点运算性能。
+ 横轴（X轴）：算术强度（Arithmetic Intensity, AI），单位：FLOP/byte（每字节数据所能执行的浮点运算次数），表示应用程序的计算密度。AI 越高，说明应用程序越倾向于计算密集型；AI 越低，则越倾向于内存密集型。
+ 水平线：计算屋顶，在 adCARM中不再是单一的绝对最大值，而是根据应用程序实际使用的指令类型（如标量、SSE、AVX、AVX512）和操作类型（如 FMA、ADD、DIV）动态调整。
+ 斜线（斜率表示带宽）：内存屋顶，表示从某一级内存（如 L1、L2、L3、DRAM）中获取数据所能支持的最大性能上限。在 adCARM 中带宽不再是理论最大值，而是根据应用程序实际使用的负载/存储比例（LD/ST ratio） 和指令数据类型（如 4B、8B、16B、32B、64B） 动态调整。
adCARM 是一个动态、应用驱动的性能模型，它通过考虑应用程序的实际指令混合、内存访问模式等特性，动态调整性能屋顶，从而提供比传统 Roofline 模型更准确、更具指导意义的性能分析和优化建议。
### 2.3、Cache-Aware Roofline Model论文里应用
| 论文 | 论文内容 |
| :-----: | :------: |
| Cache-aware Roofline model: Upgrading the loft |  本文提出了一种改进的缓存感知Roofline模型，通过更精确地建模缓存行为来提升模型的准确性。  |
| Beyond the Roofline: Cache-Aware Power and Energy-Efficiency Modeling for Multi-Cores |  本文将缓存感知模型扩展到功耗和能效领域，为多核处理器提供了更全面的性能分析。  |
| Application-driven Cache-Aware Roofline Model | 本文提出了一种以应用程序为驱动的缓存感知Roofline模型，用于更准确地分析复杂工作负载的性能瓶颈。 |
| Cache-Aware Roofline Model and Medical Image Processing Optimizations in GPUs | 本文应用缓存感知Roofline模型分析并优化了GPU上的医学图像处理算法。 |
| Exploring GPU performance, power and energy-efficiency bounds with Cache-aware Roofline Modeling | 本文利用缓存感知Roofline模型探索了GPU的性能、功耗和能效上限。 |
| Modeling Non-Uniform Memory Access on Large Compute Nodes with the Cache-Aware Roofline Model | 本文使用缓存感知Roofline模型，对大型计算节点上的非统一内存访问（NUMA）效应进行了建模和分析。 |
## 3、Hierarchical Roofline Model
### 3.1、Hierarchical Roofline Model介绍
这种模型将Roofline分析扩展到多个计算和内存层次，可以更好地分析现代异构系统（如带有GPU的CPU）的性能瓶颈。
[![pVhtTsK.png](https://s21.ax1x.com/2025/09/18/pVhtTsK.png)](https://imgse.com/i/pVhtTsK)
+ L1 Cache (一级缓存)：离处理器最近、速度最快、容量最小的“仓库”。 里面存放的通常是处理器目前正在处理、或者马上就要处理的数据。
+  L2 Cache (二级缓存)：比 L1 远一点、速度稍慢一点、但容量比 L1 大的“仓库”。 它是 L1 的“备用仓库”。当 L1 里没有处理器需要的数据时，他们就会去 L2 找。L2 的速度虽然不如 L1，但比主内存（HBM）快得多。
+ HBM (High Bandwidth Memory - 高带宽内存)：“主仓库”，容量最大，速度相对 L1 和 L2 最慢，但带宽（每秒能传输的数据量）非常高。它是GPU上最主要的内存，存储了深度学习模型的所有参数、图片数据等大量信息。当 L1 和 L2 都找不到需要的数据时，处理器就会去 HBM 中读取。

使用Hierarchical Roofline Model可以将L1、L2、HBM分层统计，以找出真正的瓶颈，做出合理的优化。
+ 找出真正的瓶颈：如果 HBM 到 L2/L1 的数据传输慢，可能说明程序需要的数据量太大，或者数据组织方式不合理，导致主内存的宽带不够用，或者数据在远距离传输上耗时太多；如果 L1/L2 的命中率低（也就是处理器经常在这些近的仓库里找不到数据，不得不去更远的 HBM 找），那说明程序的数据访问模式不好，经常访问一些不连续或不重复的数据，导致缓存效率不高。
+ 合理的优化：如果是 HBM 带宽问题，可能需要优化数据的布局，减少不必要的数据传输，或者使用更压缩的数据格式；如果是缓存命中率低，可能需要调整算法，让数据访问更有规律性，提高数据的复用率，以便数据能更多地留在 L1 和 L2 这种高速缓存中。比如，论文中提到，有些“零算术强度”的任务，它们主要在做数据搬运，可能就会在高层级内存（比如HBM）和低层级内存（L1/L2）之间来回传输，如果这类操作太多，就会影响效率。分层统计就能帮助我们发现这类问题。
### 3.2、Hierarchical Roofline Model论文里应用
| 左对齐 | 居中对齐 |
| :-----: | :------: |
| Hierarchical Roofline Performance Analysis for Deep Learning Applications |  本文提出了一种分层Roofline性能分析方法，专门用于理解和优化深度学习应用的性能。  |
| Hierarchical Roofline analysis for GPUs: Accelerating performance optimization for the NERSC‐9 Perlmutter system |  本文利用分层Roofline分析来加速NVIDIA GPU（特别是在NERSC-9 Perlmutter系统上）的性能优化。  |
| Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs | 本文详细介绍了如何利用性能工具，在Intel CPU和NVIDIA GPU上收集数据以进行分层Roofline分析。 |
## 4、Time-Based Roofline
### 4.1、Time-Based Roofline介绍
该模型将时间作为性能度量的一个维度，更适合分析深度学习等迭代性工作负载，关注完成任务所需的时间。
1、计算-带宽复杂度模型
[![pVh5bm8.png](https://s21.ax1x.com/2025/09/18/pVh5bm8.png)](https://imgse.com/i/pVh5bm8)
+ X轴：计算复杂度（Computational Complexity，单位：FLOPs）。
+ Y轴：带宽复杂度（Bandwidth Complexity，单位：Bytes）。
+ 等算术强度线（Isocurves）：对角线，斜率=1/AI。
+ 机器平衡点（Machine Balance）：硬件特定点（AI = 峰值计算/峰值带宽），区分计算/内存受限区域。
+ 开销约束框（Overhead Bound）：当复杂度乘积（FLOPs × Bytes）低于阈值时，内核启动开销主导性能。
**这张图只看任务本身需要干多少活（工作量大小和比例），不管GPU性能或花了多少时间。红点计算量大，蓝点带宽量大。**
2、计算-带宽时间模型
[![pVh5jYj.png](https://s21.ax1x.com/2025/09/18/pVh5jYj.png)](https://imgse.com/i/pVh5jYj)
+ X轴：计算时间（Compute Time，单位：秒）。
+ Y轴：带宽时间（Bandwidth Time，单位：秒）。
+ 运行时等值线（Iso-Runtime Curves）：虚线，相同运行时落在同一条线上。
+ 性能区域：
  开销受限区（Overhead Bound）：当计算/带宽时间均小于内核启动总开销；
  内存受限区（Bandwidth Bound）：带宽时间 > 计算时间；
  计算受限区（Compute Bound）：计算时间 > 带宽时间。
+ 时间定义：
 若内核计算受限：计算时间 = 实际运行时间，带宽时间 = 运行时间 × (AI / 机器平衡)；
 若内核内存受限：带宽时间 = 实际运行时间，计算时间 = 运行时间 × (机器平衡 / AI)。
3、4D复杂度-时间Roofline模型
[![pVh5zpn.png](https://s21.ax1x.com/2025/09/18/pVh5zpn.png)](https://imgse.com/i/pVh5zpn)
+ 实心符号（●/■）：表示计算/带宽复杂度。
+ 空心符号（○/□）：表示计算/带宽时间。
+ 符号间距 = 性能差距：若空心符号（实际时间）远离实心符号（理论复杂度），说明内核未达硬件极限（如因内存延迟或并行度不足）。
+ 符号重叠 ≈ 峰值性能：若实心与空心符号位置接近，表明内核充分利用硬件。

**首次将工作量复杂度（计算+带宽） 和执行时间（计算时间+带宽时间） 系统地整合到屋顶线框架中，形成“基于时间的屋顶线模型”。并且通过4D复杂度-时间图的“双符号”表示法，在一个视图中直观展示内核的工作量、执行效率（离屋顶线距离）和实际运行时间瓶颈。**
### 4.2、Time-Based Roofline论文中应用
| 论文 | 论文内容 |
| :-----: | :------: |
| Time-Based Roofline for Deep Learning Performance Analysis |  本文提出了一种时间基准的Roofline模型，用于更有效地分析深度学习应用的性能。  |
## 5、Instruction Roofline Model
### 5.1、Instruction Roofline Model介绍
传统Roofline模型只看浮点计算 (Floating-Point Operations, FLOP)。但很多现代程序（比如分析社交网络图、做基因测序）主要做的是整数运算 (Integer Operations)，甚至完全没有浮点运算，对这些程序，传统模型就失灵了。Instruction Roofline不再只看浮点运算，而是看程序执行的所有指令 (Instructions)。
[![pVhIi0U.png](https://s21.ax1x.com/2025/09/18/pVhIi0U.png)](https://imgse.com/i/pVhIi0U)
+ X轴：指令强度 (Instruction Intensity)，每条内存事务 (Transaction) 执行的平均指令数，单位为 Warp Instructions per Transaction（Instruction=总Warp级指令数/总内存事务数）。
+ Y轴：每秒执行的 Warp级指令数，单位为 GIPS (Giga Instructions Per Second)（GIPS=总Warp级指令数/内核执行时间）。
+ 各级内存带宽：还是斜线，但带宽单位换成了每秒多少亿次数据事务 (GTXN/s)。因为GPU访问不同级别的内存（比如L1缓存、L2缓存、显存HBM）速度不同，所以有多个斜线。
#### 5.1.1、内存访问模式与“墙”(Memory Walls)
1、全局内存访问模式 (Global Memory Access Patterns)
GPU的线程是成组（叫“Warp”）工作的。当一个Warp去全局内存读/写数据时，如果组内32个线程要的数据分散在内存的不同角落，GPU就需要发很多次小请求（事务）才能拿齐数据，效率很低。如果大家要的数据集中在一块连续区域，GPU发一两个大请求就能搞定，效率高。
![alt text](https://github.com/seclabBupt/aiacc/blob/main/roofline/image-1.png?raw=true)
+ Stride-0 (跨步0) 墙： 效率最高！所有32个线程都访问同一个内存地址（比如都读同一个变量）。只需要1次事务。强度最高 (32条指令/事务)。
+ Stride-1 (跨步1) 墙： 效率高！32个线程访问连续的内存地址（比如读一个数组的连续32个元素）。对于32位数据（如float, int），需要8次事务。强度中等 (4条指令/事务)。
+ Stride-8 (跨步8) 墙： 效率低！32个线程访问间隔很大（非连续）的内存地址（比如每隔8个元素读一个）。可能需要多达32次事务。强度最低 (1条指令/事务)。
2、共享内存组冲突 (Shared Memory Bank Conflicts)
GPU的共享内存（很快的小块内存，线程组内共享）被分成很多小“格子”(banks)。如果同一个Warp里的多个线程同时访问同一个bank里的不同数据，就会发生“冲突”(Bank Conflict)。冲突会导致访问被串行化（一个一个处理），拖慢速度。冲突越严重（同时访问同一个bank的线程数越多），性能越差。
![alt text](https://github.com/seclabBupt/aiacc/blob/main/roofline/image-2.png?raw=true)
+ 无冲突 (No Bank Conflict) 墙： 效率最高！32个线程访问32个不同的bank（或者同一个bank的同一个位置）。只需要1次事务。强度最高 (32条指令/事务)。
+ 32路冲突 (32-way Bank Conflict) 墙： 效率最低！32个线程都访问同一个bank里的不同位置。需要32次事务串行处理。强度最低 (1条指令/事务)。
#### 5.1.2、线程预测 (Thread Predication)
GPU线程组（Warp）执行指令是“齐步走”的。但如果组内有分支判断（比如if...else...），有些线程要走if分支，有些要走else分支。GPU怎么处理呢？它会让所有线程都走一遍所有分支的代码，但对于不该执行某个分支的线程，会用一个“掩码”(mask)把它们屏蔽掉 (Predicated Off)，让它们在那个分支里不产生实际效果（不写内存等）。这保证了硬件简单高效。但如果分支很多或者分支内工作量很大，屏蔽掉的线程就是在“空转”，浪费了计算资源。虽然Warp执行的指令数没变（Y轴GIPS可能不低），但实际干活的线程变少了，所以真正完成的有用指令数（线程级GIPS）就变低了。
![alt text](https://github.com/seclabBupt/aiacc/blob/main/roofline/image-3.png?raw=true)
+ 在指令屋顶线图上，实心点代表线程级的GIPS（有用指令吞吐）。
+ 虚线代表Warp级的GIPS（硬件执行的指令吞吐）。
+ 如果实心点紧贴虚线（如图2A），说明几乎没有线程被屏蔽，所有线程都在有效工作。
+ 如果实心点远低于虚线（如图2B），说明很多线程被屏蔽了（预测关闭），资源被浪费了，实际性能下降。

**Instruction Roofline Model通过过识别性能瓶颈是计算慢 (靠近水平屋顶)、带宽不够 (靠近某个斜线屋顶)、内存访问模式差 (靠近低效的“墙”) 还是线程闲置多 (实心点远低于虚线)，开发者可以有的放矢地进行优化。**
### 5.2、Instruction Roofline Model论文中的应用
| 论文 | 论文内容 |
| :-----: | :------: |
| An Instruction Roofline Model for GPUs |  本文介绍了一种通用的指令Roofline模型，用于分析GPU的性能。  |
| Instruction Roofline: An insightful visual performance model for GPUs | 本文将指令Roofline模型作为一种有洞察力的可视化性能模型，用于深入分析GPU的性能。|
| Metrics and Design of an Instruction Roofline Model for AMD GPUs | 本文提出了用于AMD GPU的指令Roofline模型的度量标准和设计。 |
## 6、MD-Roofline
### 6.1、MD-Roofline介绍
提出了 MD-Roofline 模型，在传统 Roofline 模型（计算 + 内存）基础上，引入通信维度，形成三维性能分析模型，可同时分析计算瓶颈、内存瓶颈和通信瓶颈。
该模型定义了系统级性能指标：
+ APF（Achievable Peak FLOPS）：系统在存在计算慢节点时可达到的峰值计算能力。
+ APM（Achievable Peak Memory bandwidth）：系统在存在内存慢节点时可达到的峰值内存带宽。
+ APB（Achievable Peak Bandwidth）：系统在存在通信慢节点时可达到的峰值通信带宽。
1、Intra-GPU Roofline（传统计算-内存平面）
![alt text](../image-4.png)
+ 横坐标（X轴）：Arithmetic Intensity（算术强度），记为 𝐴=𝐹/𝑀，单位是 FLOP/Byte。
+ 纵坐标（Y轴）：Achievable FLOPS（可达计算性能），单位是 FLOP/s。
+ 水平线：表示系统的 Achievable Peak FLOPS (APF)，即计算瓶颈的上限。
+ 斜线：斜率为 Achievable Peak Memory Bandwidth (APM)，表示内存瓶颈的上限。
+ 脊点（Ridge Point）：𝐴_𝑚𝑎𝑥=𝐴𝑃𝐹/𝐴𝑃𝑀​，是计算瓶颈和内存瓶颈的分界点。
2、Inter-GPU Communication-to-Memory Roofline（通信-内存平面）
![alt text](../image-5.png)
+ 横坐标（X轴）：Communication-to-Memory Intensity，记为 𝑂=𝑃/𝑀​，单位是（参数字节数 / 内存访问字节数）。
+ 纵坐标（Y轴）：Achievable Bandwidth（可达通信带宽），单位是 Byte/s。
+ 水平线：表示系统的 Achievable Peak Bandwidth (APB)，即通信瓶颈的上限。
+ 斜线：斜率为 APM，表示内存瓶颈对通信的限制。
+ 脊点：O_max=APB/APM​，是内存瓶颈和通信瓶颈的分界点。
+ 适用条件：当系统处于内存瓶颈区域（即 A<𝐴_𝑚𝑎𝑥​）时使用此平面。
3、Inter-GPU Communication-to-FLOPS Roofline（通信-计算平面）
![alt text](../image-6.png)
+ 横坐标（X轴）：Communication-to-FLOPS Intensity，记为 C=P/F​，单位是（参数字节数 / FLOP）。
+ 纵坐标（Y轴）：Achievable Bandwidth（可达通信带宽），单位是 Byte/s。
+ 水平线：APB，通信瓶颈上限。
+ 斜线：斜率为 Achievable FLOPS，表示计算瓶颈对通信的限制。
+ 脊点： C_max=APB/A𝑐ℎ𝑖𝑒𝑣𝑎𝑏𝑙𝑒𝐹𝐿𝑂𝑃𝑆  ，是计算瓶颈和通信瓶颈的分界点。
+ 适用条件：当系统处于计算瓶颈区域（即  A≤𝐴_𝑚𝑎𝑥  ​）时使用此平面。
### 6.2、MD-Roofline论文中应用
| 论文 | 论文内容 |
| :-----: | :------: |
| MD-Roofline: A Training Performance Analysis Model for Distributed Deep Learning |  本文提出了一种名为MD-Roofline的模型，专门用于分析分布式深度学习训练的性能瓶颈。  |
## 7、Mansard Roofline Model
### 7.1、Mansard Roofline Model介绍
论文提出的 Mansard Roofline Model（MaRM） 是对传统 Roofline 模型的一个重要扩展，旨在更准确地建模现代乱序执行处理器的性能上限。MaRM 通过引入指令退休相关的硬件约束，如退休槽位、重排序缓冲区和物理寄存器文件，来提供更真实的性能上界预测。
![alt text](../image-7.png)
+ 纵坐标（Y轴）：IPC（Instructions Retired per Cycle），即每周期退休的指令数。这与传统 Roofline 模型使用 FLOPS（每秒浮点运算次数）不同，MaRM 使用 IPC 来更直接地反映处理器的指令执行效率。
+ 横坐标（X轴）：算术强度（Arithmetic Intensity, AI），定义为：AI= 内存指令数 /非内存指令数。注意这与传统 Roofline 中 AI = FLOPS / 字节传输数 不同，MaRM 的 AI 是基于指令类型的比例。
1. 计算屋顶（Compute Roof）
+ 表示处理器在无内存瓶颈时的最大指令退休率。
+ 通常是一条水平线，高度由处理器的最大退休槽位数（RS）决定。
2. 内存屋顶（Memory Roof）
+ 表示在不同内存层级（L1、L2、L3、DRAM）下的最大可持续带宽，但以 IPC 形式表示。
+ 与传统 Roofline 的斜线不同，MaRM 的内存屋顶是非线性的“山形”曲线，因为：
+ 高延迟内存（如 DRAM）的有效带宽受限于 ROB 中并发内存请求的数量（IF_M）。
+ 带宽随并发请求数增加而增加，直到达到饱和。
3. 混合区域（Mixed Region）
+ 位于计算屋顶和内存屋顶之间。
+ 表示应用程序同时受内存带宽/延迟和计算退休率限制。
+ 在这个区域中，性能不再单纯由内存或计算决定，而是由两者共同制约。
4. 平坦区域（Flat Regions）
+ 出现在某些 AI 范围内，表示应用程序受限于退休槽位数，无法达到理论计算或内存带宽上限。
+ 例如：在 L1 屋顶附近可能出现平坦区域，表示即使 AI 很高，也无法突破退休槽位限制。
### 7.2、Mansard Roofline Model论文中应用
| 论文 | 论文内容 |
| :-----: | :------: |
| Mansard Roofline Model: Reinforcing the Accuracy of the Roofs |  本文提出了Mansard Roofline模型，旨在通过增强“屋顶线”的准确性来提高Roofline分析的精确度。  |
## 8、Multi-level Integrated Roofline Model
### 8.1、Multi-level Integrated Roofline Model介绍
多层次集成屋顶线模型 (Multi-level Integrated Roofline Model) 是对经典屋顶线模型的扩展，用于更全面地分析应用程序在不同内存层次结构中的性能瓶颈。它通过同时测量内存层次结构中多个级别（如L1缓存、L2缓存、L3缓存、MCDRAM和DRAM）的算术强度 (Arithmetic Intensity, AI) 和性能 (Performance)，来提供更深入的性能洞察。
![alt text](../image-8.png)
1、多条屋顶线（Multi-level Rooflines）：
+ 在图中，可以看到多条水平线和斜线，它们代表了不同的硬件性能上限，即“屋顶线”。
+ 水平线代表处理器在不同向量单元（如DP Vector FMA Peak, DP Vector Add Peak, Scalar Add Peak）下的理论峰值计算性能（以GFLOPS/s为单位）。这些是处理器能够达到的最高浮点运算速度。
+ 斜线代表不同内存层次结构（如L1 Cache, L2 Cache, L3 Cache/LLC, MCDRAM, DRAM）的峰值内存带宽（以GB/s为单位）。这些线条的斜率表示了在给定内存带宽下，随着算术强度的增加，应用程序能够达到的最高性能。
2、多层算术强度（Multi-level Arithmetic Intensity, AI）：
+ 传统的屋顶线模型通常只关注一个内存级别的AI，例如DRAM AI。然而，多层次集成屋顶线模型则测量并绘制应用程序在与每个内存级别交互时产生的AI。
+ 在图中，每个应用程序的特定循环或函数都由多个点表示，每个点具有相同的形状，但颜色不同。
+ 点的颜色代表了该循环/函数在不同内存层次结构中测得的算术强度：蓝色点（或红色点） 通常代表L1缓存的AI；绿色点 通常代表L2缓存的AI；黄色点 通常代表L3缓存或片上高带宽内存（如KNL中的MCDRAM）的AI；红色点 通常代表DRAM的AI。
+ 算术强度 (AI) 定义为浮点运算次数与从相应内存级别传输的字节数之比 (FLOP/Byte)。
3、性能点（Performance Points）：
+ 每个彩色的点（形状相同，颜色不同）代表一个特定的应用程序循环或函数在运行时的实际性能（y轴，GFLOPS/s）和在对应内存级别上的算术强度（x轴，FLOP/Byte）。
+ 通过观察这些点相对于不同颜色的屋顶线的位置，可以判断性能瓶颈。
**多层次集成屋顶线模型通过提供应用程序在不同内存层次结构中的详细性能视图，能够帮助开发者直观识别是计算受限还是内存带宽受限；精确定位是哪个内存级别（L1、L2、L3/LLC、MCDRAM或DRAM）的带宽成为性能瓶颈；指导优化策略，如改进缓存利用率、数据对齐、向量化或增加并行度；量化评估优化工作的效果。**
### 8.2、Multi-level Integrated Roofline Model论文中应用
| 论文 | 论文内容 |
| :-----: | :------: |
| A Novel Multi-level Integrated Roofline Model Approach for Performance Characterization |  本文提出了一种新颖的多级集成Roofline模型方法，用于对系统性能进行全面的特性化分析。  |
## 9、剩余论文总结
| 论文 | 论文内容 |
| :-----: | :------: |
| Evaluating and analyzing the energy efficiency of CNN inference on high‐performance GPU |  本文评估和分析了在高性能GPU上进行CNN推理的能效表现。  |
| RLP: Power Management Based on a Latency-Aware Roofline Model | 本文提出了一种基于延迟感知Roofline模型的功耗管理方案。 |
| A CAD-based methodology to optimize HLS code via the roofline model | 本文提出了一种基于CAD的方法，利用Roofline模型来优化高层次综合（HLS）代码。 |
| Performance and Energy-Eﬃciency Modelling for Multi-Core Processors | 本文对多核处理器的性能和能效进行了建模。|
| AIO: An Abstraction for Performance Analysis Across Diverse Accelerator Architectures | 本文提出了一种名为AIO的抽象层，用于在各种加速器架构上进行性能分析。|
| Performance analysis of deep learning workloads using roofline trajectories | 本文通过使用Roofline轨迹来分析深度学习工作负载的性能。 |
| Performance Analysis of GPU Programming Models Using the Roofline Scaling Trajectories | 本文利用Roofline扩展轨迹分析了GPU编程模型的性能。|
| On Applying Performance Portability Metrics| 本文讨论了在评估性能可移植性时如何应用相关度量标准。 |

| TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s | 本文研究了如何在TPU上实现K近邻搜索并达到峰值浮点运算性能。 |


