@Craig

# 一 前端部分

## 1. `BaseConverter`

用于处理 AI 编译器中数据的转换工作，涉及到对张量（tensor）的操作和管理。`BaseConverter` 将某种形式的模型转换为另一种形式，例如从 ONNX 格式转换为 MLIR，管理输入输出的张量、操作数、形状等。

#### 1.1 构造函数 `__init__`

```py
def __init__(self, no_save: bool = False):
    self.operands = dict()
    self.tensors = dict()
    self.shapes = dict()
    self.input_names = list()
    self.output_names = list()
    self.no_save = no_save  # do not save intermediate files in disk
```

- **`operands`**：保存转换过程中的操作数，可能是模型的某些中间数据。
- **`tensors`**：保存模型中的张量数据（例如权重），通常这些张量是 NumPy 数组。
- **`shapes`**：保存张量的形状（如 (3, 3) 或 (1, 224, 224, 3)）。
- **`input_names` 和 `output_names`**：保存输入和输出的名称，用于后续模型转换时标记数据流。
- **`no_save`**：是否保存中间文件（例如在生成 MLIR 时是否保存中间状态）。

#### 1.2 方法解析

`addShape`, `getShape`, `setShape`

这些方法用于管理张量的形状。

```py
def addShape(self, name, shape):
    ...
def getShape(self, name):
    ...
def setShape(self, name, shape):
    ...
```

- **`addShape`**：为张量添加形状，并检查是否存在冲突。
- **`getShape`**：获取某个张量的形状。
- **`setShape`**：更新某个张量的形状。



`addOperand`, `getOperand`用于管理操作数（`operand`）：

```py
def addOperand(self, name, op):
    ...
def getOperand(self, name):
    ...
```

- **`addOperand`**：将一个操作数添加到 `operands` 字典中，并检查是否存在冲突。
- **`getOperand`**：获取一个操作数。



`addWeight`, `getWeight`用于处理权重数据：

```py
def addWeight(self, name, data: np.ndarray):
    ...
def getWeight(self, name):
    ...
```

- **`addWeight`**：将权重（如模型的权重参数）添加到 `tensors` 字典中，并自动检测数据类型和形状。
- **`getWeight`**：根据名称获取权重张量。

`isWeight`, `isScalar`, `getScalar`用于检查张量是否是权重，是否是标量：

```py
def isWeight(self, name):
    ...
def isScalar(self, name):
    ...
def getScalar(self, name):
    ...
```

- **`isWeight`**：判断某个张量是否是权重（存在于 `tensors` 中）。
- **`isScalar`**：判断某个张量是否是标量（即形状为 `(1)` 或其所有元素相等）。
- **`getScalar`**：获取标量值。

`getWeightOp`用于为权重创建一个 MLIR 操作：

```py
def getWeightOp(self, name, shape: list = []):

```

- 它根据张量的类型（如 `float32`、`int32`）创建相应的 MLIR 操作，并返回。

`WeightToNpz`

将权重保存为 `.npz` 格式：

```py
def WeightToNpz(self, weight_file):
```

- 这个方法会将所有张量数据保存为一个 `.npz` 文件，便于后续的加载和使用。

## 2.`OnnxConverter`

### 2.1 三个函数

#### 2.1.1  **`translate_onnx(key, val)`**

将 ONNX 操作符的属性（`key` 和 `val`）转换为适合在当前转换过程中使用的格式。

- **`key`**：ONNX 操作的属性名称，例如 `axis`、`dtype` 等。
- **`val`**：是属性的值，通常是字符串、数字、列表等类型。

这个函数会根据不同的属性（`key`），对 `val` 进行相应的处理和转换。例如，如果 `key` 是 `axis`，会将 `val` 转换为整数

与 **`onnx_attr_translator`** 配合使用，负责将 ONNX 模型的属性进行转换，确保它们的类型符合当前的需求。

**简化的示例**：

```
def translate_onnx(key, val):
    if key == "axis":
        return int(val)
    elif key == "dtype":
        return onnx_dtype(val)
    # 可以继续根据其他属性进行处理
```

#### 2.1.2  **`onnx_dtype(dtype)`**

将 ONNX 数据类型（`dtype`）转换为对应的内部数据类型,例如，ONNX 模型中的数据类型可能是 `float32`、`int64` 等，将它们映射为 Python 或 NumPy 能够处理的类型。：

- 将 `float32` 转换为 `np.float32`
- 将 `int64` 转换为 `np.int64`

```py
def onnx_dtype(dtype):
    # 数字类型
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    # 字符串
    elif isinstance(dtype, str):
        onnx_dtype = onnx.TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

```

这里`onnx.TensorProto.DataType.Value(dtype)` 来获取相应的 ONNX 数据类型枚举值，这是 ONNX 中用于 **枚举类型转换** 的方法。`TensorProto.DataType` 是一个枚举类，它包含了所有可能的 ONNX 数据类型（例如：`FLOAT`，`INT64`，`STRING` 等）。`Value(dtype)` 方法会根据传入的字符串 `dtype` 获取对应的枚举值。例如，如果传入 `"float32"`，它会返回对应的 `DataType.FLOAT` 枚举值。



`mapping.TENSOR_TYPE_TO_NP_TYPE` 将 ONNX 数据类型映射到对应的 NumPy 类型。根据 `onnx_dtype`，通过字典查找得到相应的 NumPy 数据类型并返回。

**举例：**

```py
mapping.TENSOR_TYPE_TO_NP_TYPE = {
    onnx.TensorProto.DataType.FLOAT: np.float32,
    onnx.TensorProto.DataType.INT64: np.int64,
    onnx.TensorProto.DataType.FLOAT16: np.float16,
    # 更多的映射
}

# 假设调用的 onnx_dtype 函数
dtype = "float32"
numpy_type = onnx_dtype(dtype)

print(numpy_type)  # 输出: <class 'numpy.float32'>

```

如果 `dtype` 是 `"float32"`，`onnx.TensorProto.DataType.Value(dtype)` 会返回 `onnx.TensorProto.DataType.FLOAT`。

然后，`mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]` 会返回 `np.float32`。



#### 2.1.3 **`convert_onnx_attribute_proto(attr_proto)`**

将 ONNX 模型中的属性原型（`attr_proto`）转换为适合的 Python 对象或者数据类型。这通常是在处理 ONNX 的 `AttributeProto` 类型时使用的。

`AttributeProto` 是 ONNX 中表示属性的一种类型，可能包含标量、数组、字符串等类型的数据。这个函数将负责提取 `AttributeProto` 中的数据并转换为可以直接操作的格式，比如转换成 `numpy` 数组或者 Python 基本类型。

**简化示例**：

```
def convert_onnx_attribute_proto(attr_proto):
    # 假设 attr_proto 是某个属性的 proto 类型
    if attr_proto.HasField('i'):  # 如果是整数
        return attr_proto.i
    elif attr_proto.HasField('f'):  # 如果是浮点数
        return attr_proto.f
    elif attr_proto.HasField('strings'):  # 如果是字符串
        return attr_proto.strings
    # 可以继续处理其他类型
```

### 2.2 **三个类**

#### 2.2.1 **`BaseNode`**

`BaseNode` 是所有 ONNX 操作节点（`Node`）的基类，通常用于提供共享的属性和方法。每个 ONNX 操作都可以继承自 `BaseNode`，从而具有一些公共行为。

**示例**：

```py
class BaseNode:
    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])
        self.shape_info = dict()

# 示例：加法节点
add_node_info = {
    "name": "add1",
    "op_type": "Add",
    "attrs": {"axis": 0},
    "inputs": ["input1", "input2"],
    "outputs": ["output_add"]
}

# 示例：乘法节点
mul_node_info = {
    "name": "mul1",
    "op_type": "Multiply",
    "attrs": {"scalar": 2},
    "inputs": ["output_add"],
    "outputs": ["output_mul"]
}

# 示例：ReLU 节点
relu_node_info = {
    "name": "relu1",
    "op_type": "ReLU",
    "attrs": {},
    "inputs": ["output_mul"],
    "outputs": ["output_relu"]
}

# 创建 BaseNode 对象
add_node = BaseNode(add_node_info)
mul_node = BaseNode(mul_node_info)
relu_node = BaseNode(relu_node_info)

# 打印每个节点的信息
print(f"Node Name: {add_node.name}, Operation Type: {add_node.op_type}, Inputs: {add_node.inputs}, Outputs: {add_node.outputs}")
print(f"Node Name: {mul_node.name}, Operation Type: {mul_node.op_type}, Inputs: {mul_node.inputs}, Outputs: {mul_node.outputs}")
print(f"Node Name: {relu_node.name}, Operation Type: {relu_node.op_type}, Inputs: {relu_node.inputs}, Outputs: {relu_node.outputs}")

```

#### 2.2.2 **`OnnxNode(BaseNode)`**

`OnnxNode` 是继承自 `BaseNode` 的子类，它代表 ONNX 操作中的具体一个节点。每个节点对应 ONNX 模型中的一个操作（如 `Conv`, `MatMul` 等）。

```py
class OnnxNode(BaseNode):

    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0] # 第一个输出张量的名称
        info["op_type"] = node.op_type # 操作类型
        info["attrs"] = [(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
                         for attr in node.attribute]
        info["inputs"] = node.input # 输入
        info["outputs"] = node.output # 输出
        super().__init__(info) #调用父类的构造函数
        self.node_proto = node #存储原始的node信息
```

**`translate_onnx(attr.name, convert_onnx_attribute_proto(attr))`**：这是对每个属性值进行转换：

**`convert_onnx_attribute_proto(attr)`**：将属性的 `proto` 格式转换为 Python 对象。

**`translate_onnx(attr.name, ...)`**：将转换后的属性值进一步处理，转换为适当的类型。

```py
# 假设我们有一个 ONNX 加法节点
class OnnxAddNode:
    def __init__(self):
        self.op_type = "Add"
        self.output = ["output_add"]
        self.input = ["input1", "input2"]
        self.attribute = []  

# 创建一个简单的 ONNX Add 节点实例
onnx_add_node = OnnxAddNode()

# 将其传递给 OnnxNode
add_node = OnnxNode(onnx_add_node)

# 打印出转换后的 BaseNode 属性
print(f"Node Name: {add_node.name}")
print(f"Operation Type: {add_node.op_type}")
print(f"Inputs: {add_node.inputs}")
print(f"Outputs: {add_node.outputs}")
```

```py
# 输出
Node Name: output_add
Operation Type: Add
Inputs: ['input1', 'input2']
Outputs: ['output_add']
```

此外通过保存原始的 `node_proto`，可以在后续处理中访问 ONNX 节点的所有详细信息。



#### 2.2.3 **`OnnxConverter(BaseConverter)`**

`OnnxConverter` 是继承自 `BaseConverter` 的子类，专门用于将 ONNX 模型转换为 MLIR 或其他目标格式。

- 包含处理 ONNX 模型的特定方法， **加载 ONNX 模型**、**遍历操作节点**、**生成 MLIR 表示** 等。
- `OnnxConverter` 可以使用 `OnnxNode` 来表示 ONNX 模型中的操作节点，并将它们逐一转换为 MLIR 操作。

**简化示例**：

```
class OnnxConverter(BaseConverter):
    def __init__(self, onnx_model):
        super().__init__(no_save=False)
        self.onnx_model = onnx_model  # ONNX 模型

    def generate_mlir(self, mlir_file):
        for node in self.onnx_model.graph.node:
            onnx_node = OnnxNode(node.name, node.input, node.output, node.op_type)
            # 将 ONNX 节点转换为 MLIR 操作
            # 保存到 mlir 文件中
```



## 3、`model_transformer.py`

#### 3.1 整体内容

- Transformer 的类

	`ModelTransformer`

	`OnnxTransformer`、`TFLiteTransformer`、`CaffeTransformer` 

-  两个函数：

	- `get_model_transform(args)`：选择合适的 Transformer
	- `model_transform_func(...)`：启动转换过程的逻辑

#### 3.2  `get_model_transform`

根据传入的类型来选择转换器

```
def get_model_transform(args):
    if args.model_type == "onnx":
        return OnnxTransformer(args)
    elif args.model_type == "tflite":
        return TFLiteTransformer(args)
    ...

```

每个 transformer 都是继承 `ModelTransformer` ，实现了：

- `model_transform()`
- `model_validate()`
- `origin_inference()`（调用 ONNX/Caffe 推理）

####  3.3 各个 Transformer 类（还是以`onnx`为例）

**3.3.1 `ModelTransformer` 类：基础转换控制器**

封装了通用的模型转换逻辑：

- `self.converter`：绑定实际转换器，如 `OnnxConverter`
- `self.mlir_file`：最终生成的 MLIR 文件名
- `self.module_parsered`：MLIR 解析器 `MlirParser`，用于解析生成的 MLIR

**3.3.2  `OnnxTransformer` **

实际的转换逻辑由之前所说的 `OnnxConverter` 完成（解析 ONNX，构建 MLIR IR 节点）

**3.3.3 `mlir_opt_for_top(...)` 调用外部优化工具**

```
cmd = ["tpuc-opt", mlirfile, "-some-pass", ..., "-o", opt_mlirfile]
os.system(' '.join(cmd))
```

是将 `.mlir` 做优化的关键步骤, 调用 MLIR 工具链中的 `tpuc-opt`.



#### 3.4 model_validate

**大致判断流程**

加载输入样本 → 构建输入张量 dict

保存成 input.npz 文件

原始模型推理（由子类实现）

如果不做 MLIR 推理，直接返回

MLIR 模型推理

对比两个模型的输出 → 报告误差



## 4、model_runner 推理

负责接入原始模型（如 ONNX）与 MLIR 模型，进行前后对比推理验证，依旧是以`onnx`模型为例，比如原始`onnx`模型经过推理得到一个结果，`mlir`模型经过推理之后对结果进行对比

**4.1 整体内容**

**先看这三个函数的内容：**

 `onnx_inference(inputs, onnx_path)` `mlir_inference(inputs, mlir_path)` `final_mlir_inference`（由 `mlir_inference` 内部调用）

| 函数/类名                                            | 类型 | 功能说明                                         |
| ---------------------------------------------------- | ---- | ------------------------------------------------ |
| `onnx_inference`                                     | 函数 | 运行 ONNX 模型获取参考输出（重点）               |
| `mlir_inference`                                     | 函数 | 运行 MLIR 模型并返回输出结果                     |
| `final_mlir_inference`                               | 函数 | 内部封装的推理函数（进阶）                       |
| `_mlir_inference_by_cpu``_mlir_inference_by_cuda`    | 函数 | 选择 CPU 或 CUDA 模式跑 MLIR 模型                |
| `free_mlir_module`                                   | 函数 | 释放推理时加载的 MLIR 模块资源                   |
| `caffe_inference``tflite_inference``torch_inference` | 函数 | 用于支持其他原始模型类型（Caffe/TFLite/PyTorch） |
| `model_inference`                                    | 函数 | 封装了对 bmodel/cvimodel 的调用推理              |
| `_model_inference`                                   | 函数 | 更底层的 bmodel 推理接口                         |
| `get_cmodel_so``link_cmodel_so``link_custom_so`      | 函数 | 与 C 模型库（.so）动态链接，供后端使用           |
| `pack_bmodel_context_generator`                      | 函数 | 打包模型上下文（如 chip 类型）用于多线程         |
| `get_chip_from_model`                                | 函数 | 识别模型文件是跑在哪种芯片上                     |
| `ChipLock`                                           | 类   | 控制推理过程中芯片并发访问锁                     |
| `show_fake_cmd`                                      | 函数 | 打印提示信息（伪命令行），用于调试可视化         |

#### 4.2 onnx_inference函数

整体可以划分为以下几个内容

**4.2.1 生成dump_all_onnx**

如果需要中间层，将中间的激活值 `x.output` 添加到 `model.graph.output` 中，以便可以在 session.run() 时一并输出。

```py
for x in model.graph.node:  # 遍历所有 node
    if x.op_type in no_list:
        continue
    for name in x.output:
        if not name:
            continue
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = name
        model.graph.output.append(intermediate_layer_value_info)  # 添加为额外输出
        output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)

```



**4.2.2 准备推理session**

这里有一个先入条件，如果是 `dump_all=True`，就先把 `.onnx` 改造成带中间层输出的新文件，并替换路径。

```py
if dump_all:
    output_keys, onnx_file = generate_onnx_with_all(onnx_file)
```

创建 ONNX 推理 session，默认在 CPU 上执行。

```python
session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
```

**4.3.3 输入准备**

获取onnx输入节点

```python
inodes = session.get_inputs()
only_one = len(inputs) == 1

```

**4.3.4 执行模型推理后返回结果**

```py
outs = session.run(None, data)
```

## 5. mlir_shell

`xxx | mlir-opt | mlir-translate` 转成 Python 函数 `mlir_opt(...)` 来用，就是对命令行工具的额外封装，本质上还是用的`os.system`

主要看`mlir_opt_for_top()`的内容,前面有用到过，使用 `mlir-opt` 工具将 frontend 生成的 `.mlir` 文件优化成用于 `tpu-mlir` 后端的 MLIR。

在TPU-MLIR中将shell命令转成python的方式如下

构建shell命令列表

```
cmd = ["tpuc-opt", mlirfile]
cmd.extend([...])  # 添加各种优化 pass
cmd.extend(["-o", opt_mlirfile])
```

用`_os_system()`执行它

```py
_os_system(cmd, log_level=log_level)
```

这里_os_system()的本质就是：

```py
import os
os.system(" ".join(cmd))
```

## 6、 mlir_parser

**一句话：把MLIR文件中的内容都进来并解析成结构化数据，抽象成Python类，方便访问和处理**

两个类：Operation和MlirParser

```
MLIR 文件
   ↓
MlirParser 负责解析（读取文件、拆解结构）
   ↓
每个 Operation 行被封装成一个 Operation 对象
   ↓
所有 Operation 对象组成一个有序列表
   ↓
供后续使用：结构分析、输入输出识别、shape提取、类型判断等

```

#### 6.1 `Operation` 

用来**表示 MLIR 文件中的一个 Operation（Op）**，相当于 IR 中的一个节点。

- 记录 Op 的基本信息：名称、类型、输入输出、属性
- 方便后续按名字、类型、输入输出等方式查找或处理某个 Op

**示例：假设下面这个mlir片段**

```mlir
%0 = "top.Conv"(%input, %weight, %bias) {name = "conv1"} : ...
```

会被解析为一个operation对象，其中会包含类似信息

| 字段     | 示例                             |
| -------- | -------------------------------- |
| name     | `%0`                             |
| op_type  | `"top.Conv"`                     |
| operands | `["%input", "%weight", "%bias"]` |
| attrs    | `{"name": "conv1"}`              |

#### 6.2  `MlirParser` 

是整个模块的核心类，负责读取并解析整个 MLIR 文件生成对应的 Operation 列表，以及提取模块级别信息。

有以下大致功能

| 功能                           | 说明                                    |
| ------------------------------ | --------------------------------------- |
| `get_input_num()`              | 获取模型的输入数量                      |
| `get_batch_size()`             | 获取输入 batch 大小                     |
| `get_input_op_by_idx(i)`       | 获取第 i 个输入对应的 Operation 信息    |
| `get_op_type_by_op_name(name)` | 根据 op 名字获取它的类型                |
| `get_pre_op_by_op_name(name)`  | 获取某个 Op 的前驱 Op（用于构建依赖图） |
| `get_all_ops()`                | 返回整个 MLIR 中的所有 Operation 对象   |

**示例：**

目前有一个MLIR文件叫做model.mlir，那么可以这样用

```py
parser = MlirParser("model.mlir")
print("模型有几个输入？", parser.get_input_num())
print("第一个输入是哪个Op？", parser.get_input_op_by_idx(0))
print("模型中所有的Op类型：", [op.op_type for op in parser.get_all_ops()])

```

**整个解析过程简单介绍**

(1)`MlirParser` 读入整份 MLIR 文件；

(2)每一条形如 `%x = "xxx.Op"` 的行被识别成一个 Op；

(3)用 `Operation` 类把这些 Op 封装成结构化对象，如：

```
Operation(
    name='%1',
    op_type='top.Conv',
    operands=['%0', '%weight', '%bias'],
    attrs={'name': 'conv1'}
)
```

(4)这些 `Operation` 会被存入 `MlirParser` 的一个列表或字典里，像这样：

```py
self.ops = {
    '%0': Operation(...),
    '%1': Operation(...),
    ...
}

```

# 二、MLIR部分
# TopToTpuPass.cpp runOnOperation
## 1、初始化逻辑

**源代码：**

```c++
  module_ = getOperation();
  ctx_ = &getContext();
  mainFunc_ = module::getMainFuncOp(module_);
  LoweringConfig::isQuantized = false;
  module::setGroupQuantInfo(quantGroupSize, quantSymmetric);
  if (weightFileName != "") {
    module::setWeightFileName(weightFileName);
  }
  int64_t flops = module::getFLOPs();

```

首先需要获取当前执行的顶层操作，确定pass 被应用在哪个模块上的操作



```mlir
module {
  func.func @main() {
    ...
  }
}
```



这里整个 `module {}` 就是 `getOperation()` 返回的对象，也就是`mlir::ModuleOp module_;`



然后获取 MLIR 的全局 `MLIRContext` 上下文对象，从 module 中找到主函数 `main`，类型是 `FuncOp`。

mlir中的典型主函数：



```mlir
func.func @main(%arg0: tensor<1x3x224x224xf32>) -> () {
  ...
}
```



然后设置一个是否量化的全局标志， `LoweringConfig::isQuantized = false;`把量化的分组信息（Group Size 和是否 symmetric）写入 module 的属性中。后续如果有其它 Pass 需要知道量化设置，可以通过属性来读取这些信息，而不用全局变量。

会把类似下面的属性加到 `module` 上：



```mlir
module attributes {quant.group_size = 64, quant.symmetric = true} {
  ...
}
```



如果设置了权重文件名（比如从命令行参数传入），就把它记录在 module 属性中，供后续保存模型时使用。



**源码：**

```c++
int64_t flops = module::getFLOPs();
if (flops == 0) {
  mainFunc_.walk([&](FlopsInterface op) {
    flops += op.getFLOPs();
  });
  module::setFLOPs(flops);
}

```

如果还没有统计 FLOPs遍历一次主函数中的 Op，**自动计算 FLOPs 总量**并写入 `Module` 属性中。

通过 `getFLOPs()` **读取 module 上是否存在某个属性（如 flops 值）**。比如module IR可能有以下属性，此时则会返回12345678：

```mlir
module attributes { flops = 12345678 } {
  ...
}
```



如果当前没有统计过 FLOPs（默认为 0），就开始进行 FLOPs 的自动统计，因为有的模型可能在构建时就已经加过flops属性了。

通过 `mainFunc_.walk(...)` 遍历主函数中的所有操作（Op）；某个 Op 实现了 `FlopsInterface` 接口，就调用其 `getFLOPs()` 方法，累加到 `flops` 中，把刚刚统计出来的总 FLOPs 写回 Module 的属性中。



## 2、转换逻辑一

对 Top Dialect 中的部分 Op 应用 Tile 插入优化的 Rewrite Pattern。



```c++
  RewritePatternSet patterns(ctx_);
  patterns.clear();
  patterns.add<TryInsertTileBinaryPattern<top::SubOp>,
               TryInsertTileBinaryPattern<top::MaxOp>,
               TryInsertTileBinaryPattern<top::MinOp>,
               TryInsertTileBinaryPattern<top::CompareOp>,
               TryInsertTileMatMulPattern>(ctx_);
  if (!module::isBM1684XFamily()) {
    patterns.add<TryInsertTileBinaryPattern<top::AddOp>,
                 TryInsertTileBinaryPattern<top::MulOp>>(ctx_);
  }
  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  patterns.clear();
```

这部分就是MLIR中的Pass的知识，往 `patterns` 中添加多个 Pattern 类，

每个 Pattern 负责识别并转换一个特定的 TopOp，例如 `top.sub`、`top.max` 等。

 `TryInsertTileBinaryPattern`是一个用于匹配二元操作（如 add/sub/max/min）并尝试在其前后插入tile 操作的 Pattern。

如果当前目标芯片不是 BM1684X 系列，就多加两个 Pattern：`Add` 和 `Mul`。

通过`applyPatternsAndFoldGreedily`应用pattern到 `module_` 中的所有 Op

**转换举例：**

```
%1 = top.sub %a, %b
```

转换为：

```
%a_tiled = top.tile %a
%b_tiled = top.tile %b
%1 = top.sub %a_tiled, %b_tiled
```

然后清空pattern准备下一个pattern应用。



## 3、转换逻辑二

- **是否开启Winograd卷积算法优化**

首先判断是否开启Winograd卷积优化算法，如果 `doWinograd` 参数有值就使用它，否则默认关闭

什么是 Winograd？是一种快速卷积算法，能显著减少乘加数量，提升性能；



- **初始化量化表**

用于计算每个张量的量化参数，如 scale、zero-point、data type。可能存储的内容如下



| Tensor Name | Scale | Zero Point | Type |
| ----------- | ----- | ---------- | ---- |
| conv1_out   | 0.12  | 128        | int8 |
| input_0     | 0.007 | 0          | int8 |



- **判断当前模型是否已经量化**

检查当前模型的状态state是不是 `TOP_QUANTIZED`，判断模型是否在更早阶段就已经做了静态量化处理。

**补充模型状态：**是 `module` 里用来标记阶段的一个属性枚举，可能包括：

`TOP_F32`：浮点模型      `TOP_CALIBRATED`：已经校准（收集过 activation 范围）          `TOP_QUANTIZED`：已经量化（已经转为 int8 等整数模型)                                     `TPU_LOWERED`：已完成向 TPU dialect 的转换



在模型已经量化的情况下，设置为非对称量化（asymmetric=true），默认采用非对称模式；

在模型未量化的情况下，明确当前模型还是浮点状态，然后设置非对称量化的标志位，并执行校准流程 calibration。



- **为特定硬件执行量化逻辑**

总体结构大致划分：

```c++
if ((isBM1684X || isBM1688) && 模型未量化 && 模式为 INT8/UINT8) {
  1. MatMul 设置 per-channel 属性；
  2. MatMul + Add 设置 output_int16 属性；
  3. 检测 YOLO 模型结构并设置 int16；
}
```

**给`top::MatMulOp`设置per-channel属性**

per-channel quantization针对每个通道（channel）设置不同的 scale，适合 MatMul 和 Conv，提高精度，这个属性在 lowering 阶段被读取，从而改变生成的 MLIR/TPU 指令结构。

**给 MatMul 设置 int16 输出条件**

有些硬件不支持 float 中间结果，需要明确指示中间输出使用 int16，如果 MatMul 后面接的是 Add（加 bias），并且目标是 F16/BF16（量化的一种形式），则把中间输出设置为 int16；

**检测是否是 YOLO 结构，并设置 int16 输出**（待补充）

**针对MARS3和SGTPUV8的内容**



- **形状转换**

根据芯片型号（BM1684X 或 BM1684）来注册不同的 shape 类算子的转换规则，从 `top.shape` 等 op 转为 `tpu.shape` 相关 op

内部依然是通过 `.add<...>(ctx_)` 添加进 `RewritePatternSet patterns` 的。

**shape的转换为什么要单独拿出来？**





## 4、转换逻辑四

这里是将 Top Dialect 中的大部分算子（Op）转换为 Tpu Dialect 中的对应实现，首先依旧是按照芯片类型注册转换规则Pattern，默认 `applyPatternsAndFoldGreedily` 会 **一直应用 Pattern，直到没有任何变化为止**（可能会无限循环），这里显式地限制它最多迭代 1 次，避免某些 Pattern 自身或互相之间产生死循环。

```c++
if (module::isBM1684XFamily() || module::isBM1690Family()) { 
    bm1684x::populateTopToTpuConversionPatterns(&patterns);
  } else if (module::isBM1684Family()) {
    bm1684::populateTopToTpuConversionPatterns(&patterns);
  } else if (module::isCV18xx()) {
    cv18xx::populateTopToTpuConversionPatterns(&patterns);
  } else {
    // 如果芯片类型不匹配，则直接报错
    llvm_unreachable("Not Implemented");
  }
  // 设置 GreedyRewrite 配置，只执行一次重写
  auto config = GreedyRewriteConfig();
  // 每个 pattern 只执行一次，避免死循环
  config.maxIterations = 1; // apply each pattern only once.
  // 应用 top → tpu 的转换 pattern，进行结构重写与常量折叠
  applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
  // adjust reshape
  patterns.clear();
```



最后通过`module::updateModuleTypes()`重新整理并同步 Module 中所有 Operation 的类型信息，确保类型一致性。比如某些pattern在rewrite过程中改写了输出类型，那么需要检查每个 Operation 的输出类型。

同时将当前 Module 设置为 已完成 Top → TPU 的转换，就是一个标志位，是TPU-MLIR里的一个state枚举。

最后检查是否还有 `top::Op` 未被转换，遍历主函数中所有 Operation并且忽略掉非计算节点，对于其他的Operation则检查它所属的Dialect等等。

```c++
bool hasTopOp = false;
mainFunc_.walk([&](Operation *op) {
  if (isa<top::WeightOp, top::NoneOp, top::InputOp, ModuleOp, FuncOp, ReturnOp>(op)) {
    return;
  }
  if (!isa<tpu::TpuDialect>(op->getDialect())) {
    op->dump();
    hasTopOp = true;
  }
});
```