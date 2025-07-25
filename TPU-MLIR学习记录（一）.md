# 关于 TPU-MLIR 源码阅读（一）

@Juno

本次关于 TPU-MLIR 的源码阅读是基于在运行代码的过程中所涉及的文件的阅读，这次我们先对如下代码中的所涉及的文件进行解析。如下代码是在 TPU-MLIR 中完成模型转换的示例代码，实现的功能是将 onnx 模型变成 mlir 的模型。

```
$ model_transform \
    --model_name yolov5s \
    --model_def ../yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names 350,498,646 \
    --test_input ../image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir
```

运行完这个代码，我从日志中发现这段代码明显使用了四个文件，分别为：

- model_transform.py：主要入口文件，包括命令行参数解析（--model_name, --input_shapes 等），ONNX 模型加载与预处理，MLIR 生成和优化流程，验证测试的执行逻辑；
- tpuc-opt.cpp：MLIR 的优化部分。
- model_runner.py：模型执行的逻辑，包括 ONNX/MLIR 模型的加载和执行，输入/输出处理，NPZ 文件生成逻辑。
- npz_tool.py：包含结果比较算法。

## 1.model_transform.py

### 一、main 函数

#### 功能概述

main 函数是模型转换和验证的总入口，负责：

- 解析命令行参数
- 判断是否为 QAT（量化感知训练）模型并做特殊处理
- 根据模型类型选择合适的 Transformer 工具
- 执行模型转换（如 ONNX→MLIR）
- 可选地进行推理验证（如精度对比）
- 处理量化表的对齐与导出
- 清理中间文件

#### 主要流程

- 日志初始化：输出 TPU-MLIR 版本信息。
- 参数解析：使用 argparse 解析大量命令行参数，支持多种模型类型和转换选项。
- QAT 量化模型处理：

  - 创建 FakeQuantNodelProcessor，判断模型是否为 QAT。
  - 如果是 QAT，先做 QAT 相关的模型处理（如去除 FakeQuant 节点），并替换模型路径。
- 选择模型转换工具：调用 get_model_transform(args)，根据模型类型（onnx/caffe/tflite/torch/mlir/maskrcnn）实例化对应的 Transformer。
- 模型转换：调用 tool.model_transform(...)，执行模型转换（如 onnx→mlir）。
- 推理验证（可选）：如果设置了输入和输出，调用 tool.model_validate(...)，对转换前后模型做推理并比对精度。
- QAT 量化表对齐与导出：如果是 QAT，调用 align_final_opt 和 align_canonicalize，对齐量化表与实际模型节点名。
- 记录和清理

  - 保存转换过程中的文件记录。
  - 非 debug 模式下清理中间文件。

### 二、main 函数的调用函数

#### 1. FakeQuantNodelProcessor 相关

- 功能：判断并处理 QAT 量化模型，去除 FakeQuant 节点，生成量化表。
- 主要方法：

  - process_model()：对 QAT onnx 模型做后处理，生成 _qat.onnx。_
  - align_final_opt()、align_canonicalize()：对齐量化表与实际模型节点名。

#### 2. get_model_transform(args)

- 功能：根据模型类型（onnx/caffe/tflite/torch/mlir/maskrcnn）实例化对应的 Transformer 工具。
- 逻辑：

  - 判断模型后缀，选择合适的 Transformer 子类（如 OnnxTransformer、CaffeTransformer 等）。
  - 每个 Transformer 封装了模型转换、推理、验证等功能。

#### 3. tool.model_transform(...)

- 功能：执行模型转换（如 onnx→mlir）。
- 逻辑：

  - 生成 origin mlir（如 onnx→mlir）。
  - 调用 mlir 优化工具（如 mlir_opt_for_top）做结构优化和后处理。
  - 记录转换过程和文件。

#### 4. tool.model_validate(...)

- 功能：对转换前后模型做推理并比对精度。
- 逻辑：

  - 加载输入数据（npz/jpg/npy）。
  - 调用原始模型和 mlir 模型分别推理，保存输出。
  - 调用 f32_blobs_compare 对比输出精度。
  - 记录推理过程和文件。

#### 5. tool.file_recorder.dump()、tool.cleanup()

- 功能：记录转换过程中的文件，清理中间文件。

### 三、基础功能函数

#### 1. ModelTransformer 及其子类

##### ModelTransformer

- 功能：模型转换的基类，定义了转换、推理、验证等通用接口。
- 主要方法：

  - model_transform：模型转换主流程。
  - model_validate：推理验证主流程。
  - origin_inference：原始模型推理（抽象方法，由子类实现）。

##### 子类（OnnxTransformer、CaffeTransformer、TFLiteTransformer、TorchTransformer、MlirTransformer、MaskRCNNTransformer）

- 功能：针对不同模型格式实现具体的转换和推理逻辑。
- 主要方法：

  - origin_inference：调用对应的推理后端（如 onnx_inference、caffe_inference 等）。

#### 2. ensure_batch_size

- 功能：保证输入张量的 batch size 满足要求（如自动扩展）。
- 逻辑：如果 batch size 不足，则重复填充。

#### 3. model_mlir

- 功能：生成原始 mlir 文件。
- 逻辑：调用 converter 的 generate_mlir 方法。

#### 4. model_transform

- 功能：模型转换主流程。
- 逻辑：

  - 生成 origin mlir。
  - 调用 mlir 优化工具做结构优化。
  - 记录转换过程和文件。

#### 5. model_validate

- 功能：推理验证主流程。
- 逻辑：

  - 加载输入数据。
  - 分别调用原始模型和 mlir 模型推理。
  - 对比输出精度。
  - 记录推理过程和文件。

## 2.BaseConverter.py

### 一、整体定位

该文件定义了一个名为 BaseConverter 的基类，主要用于张量（tensor）、操作数（operand）、形状（shape）等在模型转换过程中的管理。它为后续的具体转换器（如 ONNX、Caffe、TensorFlow 转 MLIR 等）提供了基础的数据结构和通用方法。

### 二、主要成员变量

```python
def __init__(_self_, _no_save_: bool = False):
        _self_.operands = dict()
        _self_.tensors = dict()
        _self_.shapes = dict()
        _self_.input_names = list()
        _self_.output_names = list()
        _self_.no_save = _no_save_ _# do not save intermediate files in disk_
```

- self.operands：字典，保存操作数（operand）的名称与对象的映射。
- self.tensors：字典，保存权重（weight）或常量张量的名称与 numpy 数组的映射。
- self.shapes：字典，保存张量名称与其 shape（list）的映射。
- self.input_names / self.output_names：输入、输出张量的名称列表。
- self.no_save：布尔值，是否不保存中间文件。

### 三、主要方法功能梳理

#### 1. 形状相关

```python
def addShape(self, name, shape):
        if len(shape) == 0:
            shape = [1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        if name in self.shapes:
            if self.shapes[name] != shape:
                raise KeyError("shape {} conflict {} vs {}".format(name, self.shapes[name], shape))
        self.shapes[name] = shape

    def getShape(self, name):
        if name not in self.shapes:
            raise KeyError("shape {} not found".format(name))
        return self.shapes[name]

    def setShape(self, name, shape):
```

- addShape(name, shape):添加 shape 信息，支持 tuple、list，若已存在则检查一致性。
- getShape(name):获取指定名称的 shape，不存在则报错。
- setShape(name, shape):修改已存在的 shape，不存在则报错。

#### 2.操作数相关

```python
def addOperand(self, name, op):
        if name in self.operands:
            if self.operands[name] != op:
                raise KeyError("operand {} conflict".format(name))
            return
        self.operands[name] = op

    def getOperand(self, name):
        if name not in self.operands:
            raise KeyError("operand {} not found".format(name))
        return self.operands[name]

    def getOp(self, name):
        if self.isWeight(name):
            return self.getWeightOp(name)
        return self.getOperand(name)
```

- addOperand(name, op):添加操作数，若已存在则检查一致性。
- getOperand(name):获取操作数对象，不存在则报错。
- getOp(name):如果是权重，返回权重操作数，否则返回普通操作数。

#### 3. 权重/张量相关

```python
def addWeight(_self_, _name_, _data_: np.ndarray):
        _if_ not isinstance(_data_, np.ndarray):
            _raise_ KeyError("tensor data must be a numpy array")
        _if_ _data_.dtype != np.float32:
            _data_ = _data_.astype(np.float32)
        _if_ _name_ in _self_.tensors:
            _if_ np.all(_self_.tensors[_name_] == _data_):
                _return_
            _raise_ KeyError("tensor {} conflict".format(_name_))
        _if_ len(_data_.shape) == 0:
            _data_ = _data_.reshape([1])
        _# all weight convert to f32._
        _self_.tensors[_name_] = _data_
        _self_.addShape(_name_, _data_.shape)

    def isWeight(_self_, _name_):
        _if_ _name_ in _self_.tensors:
            _return_ True
        _return_ False

    def getWeight(_self_, _name_):
        _if_ _name_ not in _self_.tensors:
            _raise_ KeyError("No {} tensor in model".format(_name_))
        _return_ _self_.tensors[_name_]
        
    def getWeightOp(_self_, _name_, _shape_: list = []):
        _if_ _name_ not in _self_.tensors:
            _raise_ KeyError("Should addWeight first:{}!!!".format(_name_))
        old_shape = _self_.getShape(_name_)
        _if_ _shape_ and old_shape != _shape_:
            _assert_ (np.prod(old_shape) == np.prod(_shape_))
            old_shape = _shape_
        ori_type = str(_self_.tensors[_name_].dtype)
        type_dict = {
            'int8': "INT8",
            'uint8': "UINT8",
            'float32': "F32",
            'int32': "INT32",
            'int16': "INT16",
            'uint16': "UINT16",
        }
        _if_ ori_type not in type_dict:
            _raise_ KeyError("type {} not implemented".format(ori_type))
        op = _self_.mlir.create_weight_op(_name_, old_shape, type_dict[ori_type])
        _self_.addOperand(_name_, op)
        _return_ op
```

- addWeight(name, data):添加权重张量，要求为 numpy 数组，自动转为 float32，shape 也会同步添加。
- isWeight(name):判断是否为权重（即是否在 tensors 字典中）。
- getWeight(name):获取权重张量，不存在则报错。
- getWeightOp(name, shape=[]):获取权重的操作数对象（通常用于 MLIR），必要时会 reshape，类型映射到 MLIR 支持的类型。

```python
def isScalar(_self_, _name_):
        _if_ not _self_.isWeight(_name_): _return_ False
        _if_ np.prod(_self_.getShape(_name_)) == 1: _return_ True
        w = _self_.getWeight(_name_)
        _return_ np.all(w == w.flatten()[0])

    def isScalar_(_self_, _name_, _x_):
        _assert_ (isinstance(_x_, (int, float)))
        _if_ not _self_.isWeight(_name_): _return_ False
        _if_ np.prod(_self_.getShape(_name_)) == 1: _return_ True
        w = _self_.getWeight(_name_)
        _return_ np.all(w == _x_)

    def getScalar(_self_, _name_):
        _if_ not _self_.isScalar(_name_):
            _raise_ RuntimeError("Not Scalar")
        _return_ _self_.getWeight(_name_).flatten()[0]
```

- isScalar(name):判断是否为标量（shape 乘积为 1，且所有值相等）。
- isScalar_(name, x):判断是否为值为 x 的标量。
- getScalar(name):获取标量的值。

#### 4. 其他

```python
def generate_mlir(_self_, _mlir_file_: str):
        _raise_ NotImplementedError('generate_mlir')
        
    def WeightToNpz(_self_, _weight_file_):
        tensor_npz = {}
        _for_ name _in_ _self_.tensors:
            _if_ name in _self_.operands:
                tensor_npz[name] = _self_.tensors[name]
        np.savez(_weight_file_, **tensor_npz)
```

- generate_mlir(mlir_file):生成 MLIR 文件的接口，基类未实现，需子类实现。
- WeightToNpz(weight_file):将所有权重（且已添加为操作数的）保存为 npz 文件。

---

### 四、典型的使用流程（举例说明）

假设现在要把一个深度学习模型（比如 ONNX、Caffe、TensorFlow）转换成 MLIR 格式，大致会经历以下几个步骤：

#### 步骤一：收集和注册模型信息

- 读取模型文件，遍历每一层（layer）或每个节点（node）。
- 对于每个输入、输出、权重、常量等，调用如下方法注册到 converter 里：
- addShape(name, shape)：记录每个张量的形状（比如 [1, 3, 224, 224]）。
- addWeight(name, data)：记录每个权重的具体数值（numpy 数组），比如卷积的权重。
- addOperand(name, op)：记录每个操作数（比如某一层的输出）。

#### 步骤二：在转换过程中随时查询

- 需要用到某个张量/权重/操作数时，调用 getShape(name)/getWeight(name)/getOperand(name)这些方法保证你能随时拿到你需要的数据。

#### 步骤三：判断和处理特殊情况

- 比如有些权重其实是“标量”（只有一个数），可以用 isScalar(name)/getScalar(name)判断和获取。

#### 步骤四：生成 MLIR 文件

- 当所有信息都收集好后，调用 generate_mlir(mlir_file)（子类实现）来生成最终的 MLIR 文件。

#### 步骤五：保存权重

- 如果需要把权重单独保存成 npz 文件，调用 WeightToNpz(weight_file)。

## 3.deploy_qat.py

### 一、整体定位

本文件主要用于处理 QAT（Quantization Aware Training，量化感知训练）后的 ONNX 模型，对模型中的 FakeQuant 节点进行处理、裁剪、参数提取、表格导出等操作，为后续的部署和推理做准备。核心类为 FakeQuantNodelProcessor，它负责识别、处理、移除 QAT 相关节点，并生成量化表（calitable、qtable）等。下面是阅读这个文件的代码的简化版的结构。

```
FakeQuantNodelProcessor
    |
    |-- process_model()
            |-- prepare_params()
            |-- remove_fake_pad_op()
            |-- 遍历节点
                    |-- deal_with_weight_fakequant()
                    |-- clip_weight()
                    |-- parse_qparams()
                    |-- deal_with_activation_fakequant()
            |-- 移除节点
            |-- onnx.shape_inference.infer_shapes
            |-- 移除未用initializer
            |-- post_process_clip_ranges()
    |
    |-- export_tables()
    |-- align_final_opt() / align_canonicalize()  # 名字对齐
```

### 二、主要函数与成员变量

#### 1. 量化节点类型定义

```sql
PERCHANNEL_FAKEQUANTIZER = ['FakeQuantizeLearnablePerchannelAffine',
                                'FixedPerChannelAffine',
                                'FakeQuantizeDSQPerchannel',
                                'FPEmuOp_per_channel']
    PERTENSOR_FAKEQUANTIZER = ['LearnablePerTensorAffine',
                               'FixedPerTensorAffine',
                               'FakeQuantizeDSQPertensor',
                               'FakeQuantizeTqtAffine',
                               'FPEmuOp_per_tensor']
    ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER
```

- PERCHANNEL_FAKEQUANTIZER：按通道量化的 FakeQuant 节点类型列表。
- PERTENSOR_FAKEQUANTIZER：按张量量化的 FakeQuant 节点类型列表。
- ALL_FAKEQUANTIZER：所有支持的 FakeQuant 节点类型。

#### 2. 初始化与成员变量

```python
def __init__(_self_, _input_model_path_, _input_model_name_):
        _self_.input_model_name = _input_model_name_
        _self_.input_model_path = _input_model_path_
        _self_.output_model_path = _input_model_path_.replace('.onnx', '_qat.onnx')
        _self_.fakequant_model = _self_.check_onnx_for_fakequant_nodes(_input_model_path_)
        _self_.calitable_name = _input_model_name_ + "_calitable_qat"
        _self_.qtable_name = _input_model_name_ + "_qtable_qat"
        _self_.nodes_to_be_removed = []
        _self_.cali_table = {}
        _self_.weight_table = {}
        _self_.q_table = {}
```

- input_model_path/input_model_name：输入模型路径和名称。
- output_model_path：输出模型路径（自动加后缀）。
- fakequant_model：模型中是否包含 FakeQuant 节点。
- calitable_name/qtable_name：量化表文件名。
- nodes_to_be_removed：待移除节点列表。
- cali_table/weight_table/q_table：量化参数表、权重量化表、量化类型表。

#### 3.process_model()

process_model() 是 FakeQuantNodelProcessor 类的核心方法，主要用于对 QAT（量化感知训练）得到的 ONNX 模型进行后处理。其目标是：

- 移除 FakeQuant 节点（量化伪节点）
- 修正权重数据
- 生成量化和校准表
- 简化模型结构，便于后续部署

```python
def process_model(_self_):
        _self_.prepare_params() #初始化输入输出映射、参数、initializer。
        _self_.remove_fake_pad_op() #移除无效 Pad 节点，修正后续节点输入。
        _self_.update_inp2node_out2node(_self_.graph) #建立输入到节点、输出到节点的映射。
        _for_ node _in_ _self_.graph.node:
            print(f'process_node :{node.name}, type:{node.op_type}')
            # 处理伪量化节点
            _if_ node.op_type in _self_.ALL_FAKEQUANTIZER:
                _self_.nodes_to_be_removed.append(node)
                _self_.nodes_to_be_removed.extend(_self_.get_constant_inputs(node))
            # 处理输出节点映射
            _if_ node.output[0] not in _self_.inp2node:
                _assert_ node.output[0] in [l.name _for_ l _in_ _self_.graph.output]#[l.name for l in self.graph.output]：提取模型所有最终输出的名称，用于验证 “未被映射的输出” 的合法性。
                _self_.inp2node[node.output[0]] = []
            next_nodes = _self_.inp2node[node.output[0]]
            #权重量化节点处理（PERCHANNEL_FAKEQUANTIZER）：调用 deal_with_weight_fakequant，将权重 FakeQuant 节点替换为原始权重，收集冗余节点。
            #调用 clip_weight，对权重做裁剪。
            #调用 parse_qparams，解析量化参数，生成权重量化表和量化类型表。
            _if_ node.op_type in _self_.PERCHANNEL_FAKEQUANTIZER:
                redundant_nodes = _self_.deal_with_weight_fakequant(node)
                _self_.nodes_to_be_removed.extend(redundant_nodes)
                _self_.clip_weight(node)
                # 生成权重量化参数表
                tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = _self_.parse_qparams(node)
                _if_ len(next_nodes) == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    next_node_output = next_nodes[0][0].output[0]
                    _assert_ next_nodes[0][0].op_type in ['Gemm', 'Conv']
                    _if_ _self_.inp2node[next_node_output][0][0].op_type == 'Relu':
                        tensor_name = '{}_{}_weight'.format(_self_.inp2node[next_node_output][0][0].output[0], 'Relu')
                    _else_:
                        tensor_name = '{}_{}_weight'.format(next_node_output, next_nodes[0][0].op_type)
                    _self_.weight_table[tensor_name] = [
                        tensor_name,
                        len(scale),
                        *[float(f"{float(x):.7f}") _for_ x _in_ scale],
                        len(zero_point),
                        *[int(x) _for_ x _in_ zero_point]
                        ]
                    _self_.q_table[tensor_name] = [tensor_name,quant_type]
             # 处理张量级伪量化（激活或权重）
            _elif_ node.op_type in _self_.PERTENSOR_FAKEQUANTIZER:
                _if_ len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    _# 处理权重伪量化_
                    redundant_nodes = _self_.deal_with_weight_fakequant(node)
                    _self_.nodes_to_be_removed.extend(redundant_nodes)
                    _self_.clip_weight(node)
                    tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = _self_.parse_qparams(node)
                    _assert_ next_nodes[0][0].op_type in ['Gemm', 'Conv']
                    tensor_name_new = '{}_{}_weight'.format(next_nodes[0][0].output[0], next_nodes[0][0].op_type)
                    _self_.weight_table[tensor_name_new] = [
                        tensor_name_new,
                        len(scale),
                        *[float(f"{float(x):.7f}") _for_ x _in_ scale],
                        len(zero_point),
                        *[int(x) _for_ x _in_ zero_point]
                        ]
                    _self_.q_table[tensor_name_new] = [tensor_name_new,quant_type]
                _else_:
                    # 处理激活伪量化
                    tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = _self_.parse_qparams(node)
                    tensor_name = node.input[0]
                    _if_ node.input[0] in _self_.out2node:
                        pre_node = _self_.out2node[tensor_name]
                        pre_type = pre_node.op_type
                        tensor_name = '{}_{}'.format(tensor_name, pre_type)
                    _self_.cali_table[tensor_name] = [
                        tensor_name,
                        float(f"{float(scale * max(-qmin, qmax)):.7f}"),
                        float(f"{float(scale * (qmin - zero_point)):.7f}"),
                        float(f"{float(scale * (qmax - zero_point)):.7f}")
                    ]
                    _self_.q_table[tensor_name] = [tensor_name,quant_type]
                    _self_.deal_with_activation_fakequant(node)
                    output_name = node.output[0]
                    _for_ out _in_ _self_.graph.output:
                        _if_ out.name == output_name:
                            out.name = node.input[0]
        #移除冗余节点
        _for_ node _in_ _self_.nodes_to_be_removed:
            _self_.graph.node.remove(node)
        #形状推理与保存模型
        model_onnx = onnx.shape_inference.infer_shapes(_self_.model)
        onnx.save(model_onnx, _self_.output_model_path)

        _# 清理初始化器_
        _self_.out2node, _self_.inp2node = _self_.update_inp2node_out2node(_self_.graph)
        _self_.named_initializer = _self_.prepare_initializer(_self_.graph)
        _for_ name, initial_data _in_ _self_.named_initializer.items():
            _if_ name in (_self_.out2node.keys() | _self_.inp2node.keys()):
                _continue_
            _self_.graph.initializer.remove(initial_data)
        #后处理裁剪范围
        _self_.post_process_clip_ranges()
```

- prepare_params():初始化输入输出映射、参数、initializer。
- remove_fake_pad_op():移除无效 Pad 节点，修正后续节点输入。
- update_inp2node_out2node(graph)：建立输入到节点、输出到节点的映射。
- deal_with_weight_fakequant(node)：处理权重量化的 FakeQuant 节点，将其替换为原始权重，返回冗余节点。
- clip_weight(node)：对权重做裁剪（clip）到量化区间。
- parse_qparams(node)：解析量化参数（scale、zero_point、qmin、qmax、dtype、quant_type）。
- deal_with_activation_fakequant(node)：处理激活量化的 FakeQuant 节点，将其前后节点直接连接。
- post_process_clip_ranges()：对特殊节点（如 Flatten、Reshape 等）输入输出的 clip range 做对齐，补全校准表。
- numpy_helper.to_array ()： 是 ONNX（Open Neural Network Exchange）工具包中的一个函数，其主要功能是将 ONNX 格式的张量（TensorProto）转换为 NumPy 数组。

#### 4.clip_weight(_self_, _node_)

该函数的核心作用是将权重裁剪到量化对应的浮点范围内，确保量化时不会因数值溢出导致精度损失。

```python
def clip_weight(self, node):
    # 1. 从伪量化节点中解析量化参数
    # tensor_name：权重名称；scale/zero_point：量化缩放和零点；qmin/qmax：量化范围边界
    # dtype：数据类型；quant_type：量化类型（如PER_CHANNEL/PER_TENSOR）
    tensor_name, scale, zero_point, qmin, qmax, dtype, quant_type = self.parse_qparams(node)
    
    # 2. 根据权重名称获取原始权重数据（NumPy数组格式）
    data = self.name2data[tensor_name]
    
    # 3. 计算量化对应的浮点裁剪范围
    # 公式：浮点最小值 = (量化最小值 - 零点) * 缩放因子
    # 作用：将整数量化范围[qmin, qmax]转换为对应的浮点数范围
    clip_range_min = ((qmin - zero_point) * scale).astype(data.dtype)
    clip_range_max = ((qmax - zero_point) * scale).astype(data.dtype)
    
    # 4. 判断是否为通道级量化（每个通道有独立的scale）
    # scale.shape[0] > 1 表示每个通道有独立的缩放因子
    if len(scale.shape) > 0 and scale.shape[0] > 1:
        new_data = []  # 存储裁剪后的权重
        transposed = False  # 标记是否需要转置（针对反卷积）
        
        # 5. 获取当前节点的下游节点（使用该权重的节点）
        next_node = self.inp2node[node.output[0]]
        
        # 6. 特殊处理反卷积层（ConvTranspose）的权重维度
        # 反卷积权重维度通常为[in_channels, out_channels, h, w]，需转置为[out_channels, ...]
        if len(next_node) == 1 and next_node[0][0].op_type == 'ConvTranspose':
            transposed = True
            data = data.transpose(1, 0, 2, 3)  # 调整维度顺序以匹配通道
        
        # 7. 按通道裁剪权重（每个通道使用独立的裁剪范围）
        for c in range(data.shape[0]):
            # 对第c个通道的权重裁剪到该通道对应的范围
            new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
        
        # 8. 将列表转换为NumPy数组
        new_data = np.array(new_data)
        
        # 9. 若之前转置过，恢复反卷积权重的原始维度
        if transposed:
            new_data = new_data.transpose(1, 0, 2, 3)
    
    # 10. 若为张量级量化（所有通道共享一个scale），直接裁剪整个权重
    else:
        new_data = np.clip(data, clip_range_min, clip_range_max)
    
    # 11. 将裁剪后的NumPy数组转换为ONNX张量格式
    new_data = numpy_helper.from_array(new_data)
    
    # 12. 更新初始化器中的权重数据（替换为裁剪后的数据）
    self.named_initializer[tensor_name].raw_data = new_data.raw_data
```

#### 5.parse_qparams(self, node)

`parse_qparams` 是 QAT（量化感知训练）模型后处理中的量化参数解析核心函数。它的核心作用是从伪量化（`FakeQuant`）节点中提取关键量化参数（如缩放因子、零点、量化范围等），并自动判断量化类型（如 INT8、UINT8、FP8），为后续模型量化部署（如权重裁剪、算子融合、硬件适配）提供必要的参数支持。

##### 提取关键量化参数

在计算图（如 ONNX）中，“输入（Input）” 和 “属性（Attribute）” 的设计有明确分工：

- 输入（Input）：用于传递 “动态数据”（如张量、中间计算结果），这些数据在图执行过程中可能变化（如不同批次的输入、动态生成的量化参数）。
- 属性（Attribute）：用于传递 “静态配置”（如固定的数值、类型），这些参数在图定义时就已确定，执行过程中不会变化（如 INT8 的固定范围 [-128,127]）。

FakeQuant 节点的核心参数（qmin/qmax、scale、zero_point）既可能是 “动态生成的”（需通过输入传递），也可能是 “固定配置的”（需通过属性传递），因此代码需要兼容两种情况。具体原因如下：

1. 输入传入（qmin/qmax 在 node.input 中）的场景与设计逻辑

适用场景：当 qmin/qmax 等参数是 “动态计算的结果” 时（如通过校准算法生成的量化范围）。

例如：在模型训练或校准阶段，qmin/qmax 可能根据输入数据的分布动态计算（如通过 “最小最大校准法” 统计输入的最小值和最大值作为 qmin/qmax），这些动态生成的参数会作为 “张量数据” 通过节点的 “输入” 传递给 FakeQuant 节点。

```python
if len(node.input) > 3:# 情况1：qmin/qmax通过节点输入传入（部分FakeQuant节点设计）
        qmin, qmax = node.input[-2:]
        qmin, qmax = self.name2data[qmin], self.name2data[qmax]  # 转换为数值# 从属性中提取数据类型（如“int8”）if len(node.attribute) > 0:
            qparams = self.parse_attrs(node.attribute)  # 解析属性为字典
            dtype = qparams['dtype']else:
            dtype = 'None'
```

这里的 `node.input` 是一个列表，存储了节点的所有输入张量名称。当输入长度超过 3 时（前 3 个通常是 tensor_name、scale、zero_point），最后 2 个输入就是动态生成的 qmin 和 qmax（以张量名称的形式存储，需通过 `self.name2data` 映射到实际数值）。

1. 属性传入（qmin/qmax 在 node.attribute 中）的场景与设计逻辑

适用场景：当 qmin/qmax 等参数是 “固定配置” 时（如预定义的量化类型对应的固定范围）。
例如：INT8 的量化范围是固定的 `qmin=-128，qmax=127`，UINT8 是 `qmin=0，qmax=255`，这些无需动态计算，可直接作为 “静态配置” 写入节点属性。

```python
elif len(node.attribute) > 0:# 情况2：qmin/qmax和dtype通过节点属性传入（另一类FakeQuant节点设计）
        qparams = self.parse_attrs(node.attribute)  # 解析属性（如quant_min、quant_max）
        qmin = qparams['quant_min']  # 量化最小值（如INT8的-128）
        qmax = qparams['quant_max']  # 量化最大值（如INT8的127）
        dtype = qparams['dtype']  # 数据类型（如“int”“uint”）
    else:# 异常情况：未找到qmin/qmax（可能导致后续量化错误）
        print.info(f'qmin and qmax are not found for <{node.name}>!')
```

这里的 `node.attribute` 是节点的静态配置列表，存储了预定义的量化范围（如 `quant_min` 对应 qmin，`quant_max` 对应 qmax），这些参数在图定义时就已确定，无需动态计算。

##### 自动判断量化类型

```python
_if_ qmax == float(448.0) or qmax == float(57344.0):
            quant_type = 'FP8'
```

- FP8（浮点 8 位）是一种特殊的量化格式，它的量化范围由硬件标准定义，而非整数范围。目前主流的 FP8 标准（如 IEEE 754-2019）中：

  - `qmax=448.0` 对应 FP8 的 “E4M3” 格式（指数位 4 位，尾数位 3 位）的最大正值范围；
  - `qmax=57344.0` 对应 FP8 的 “E5M2” 格式（指数位 5 位，尾数位 2 位）的最大正值范围。
- 逻辑：通过 `qmax` 的固定值直接匹配 FP8 类型（无需计算位数，因 FP8 不是整数离散范围）

```python
bit = int(np.log2(qmax-qmin+1))
        _if_ (bit == 8 and qmin < 0):
            dtype = 'int'
            quant_type = 'INT8'
        _elif_ (bit == 8 and qmin ==0):
            dtype = 'uint'
            quant_type = 'UINT8'
        _elif_ (bit == 4 and qmin < 0):
            dtype = 'int'
            quant_type = 'INT4'
```

整数量化类型（如 INT8、INT4）的量化范围是 “连续整数”，范围大小（`qmax - qmin + 1`）正好等于 `2^bit`（bit 为量化位数）。

位数（bit）：决定是 8 位还是 4 位量化（通过 `bit` 值判断）；

符号（qmin）：决定是带符号（`qmin < 0`，如 INT8、INT4）还是无符号（`qmin = 0`，如 UINT8）。

### 三、其他函数

- __init__(self, input_model_path, input_model_name)：初始化类成员变量，加载模型，设置输出路径、表名等。
- check_onnx_for_fakequant_nodes(self, onnx_model_path)：检查 ONNX 模型中是否包含 FakeQuant 节点。
- prepare_params(self)：准备模型的输入输出映射、参数、initializer。
- parse_attrs(self, node_attrs)：解析节点属性，转为 Python 字典。
- update_inp2node_out2node(self, graph)：建立输出到节点、输入到节点的映射。
- prepare_data(self, graph)：收集所有 initializer 和 Constant 节点的数据。
- prepare_initializer(self, graph)：收集所有 initializer 节点。
- get_constant_inputs(self, node)：获取节点的常量输入节点。
- remove_fake_pad_op(self)：移除所有 pads 全为 0 的 Pad 节点，并修正后续节点输入。
- deal_with_weight_fakequant(self, node)：处理权重量化的 FakeQuant 节点，将其替换为原始权重，返回冗余节点。
- weight_preprocess(self, target_tensor)：对权重做归一化处理（tanh+ 归一化），并找出冗余节点。
- get_constant_inputs(self, node)：获取节点的常量输入节点。
- post_process_clip_ranges(self)：对 Flatten、Resize、Reshape、Transpose 等节点的输入输出做 clip range 对齐，补全校准表。
- deal_with_activation_fakequant(self, node)：处理激活量化的 FakeQuant 节点，将其前后节点直接连接。
- export_tables(self)：导出校准表（calitable）和量化表（qtable）。
- get_activation_mappings(self, model)：获取激活节点的输入输出映射。
- align_final_opt(self, node_name_mapping, onnx_sim="")：对齐优化后的 ONNX 模型和原始模型的激活节点名。
- creat_new_qat_item(self, oir_name, new_name)：在表格中创建新的 QAT 项（用于节点名对齐）。
- align_canonicalize(self, mlir_file, test_result)：对齐 MLIR 文件中的激活名和 QAT 表。
- extract_activation_names_mlir(self, mlir_path)：从 MLIR 文件中提取激活名。
- compare_npz_name_mapping(self, test_result)：通过 npz 文件比对激活名，辅助对齐。

## 4.model_runner.py

### 一、整体定位

这个文件是一个多框架模型推理工具，支持 ONNX、MLIR、TFLite、Caffe、Torch、BModel 等多种格式的模型推理，并能将推理结果保存为 npz 文件。它既可以作为命令行工具使用，也可以作为 Python 模块被调用。主要功能包括：

- 加载输入数据（npz）
- 根据模型类型选择合适的推理后端
- 支持 dump 所有中间张量
- 支持多种硬件/仿真环境
- 支持多线程/多进程下的硬件资源锁管理

### 二、main 函数的整体流程

1. 参数解析：使用 argparse 统一解析命令行参数，获取输入文件、模型文件、输出文件、模型类型等信息。
2. 加载输入数据：通过 np.load(args.input) 加载输入的 npz 文件，得到输入张量字典。
3. 根据模型类型分发推理函数：通过判断模型文件的后缀，选择合适的推理函数进行模型推理。

   - .final.mlir → final_mlir_inference
   - .mlir → mlir_inference
   - .onnx → onnx_inference
   - .tflite → tflite_inference
   - .prototxt + .caffemodel → caffe_inference
   - .pt/.pth → torch_inference
   - .bmodel/.cvimodel → model_inference
   - 其他 → 报错
4. 保存推理结果

```python
_if_ __name__ == '__main__':
    _# yapf: disable_
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", _required_=True, _help_="input npz file")
    parser.add_argument("--model", _type_=str, _required_=True,
                        _help_="mlir/pytorch/onnx/tflie/bmodel/prototxt file.")
    parser.add_argument("--weight", _type_=str, _default_="",
                        _help_="caffemodel for caffe")
    parser.add_argument("--output", _default_='_output.npz',
                        _help_="output npz file")
    parser.add_argument("--dump_all_tensors", _action_='store_true',
                        _help_="dump all tensors to output file")
    parser.add_argument("--debug", _type_=str, _nargs_="?", _const_="",
                        _help_="configure the debugging information.")
    parser.add_argument("--out_fixed", _action_="store_true",
                        _help_="no float number transforming, only for int8/uint8.")
    parser.add_argument("--cuda", _action_="store_true",
                        _help_="use cuda to do inference")
    parser.add_argument("--decrypt_lib", _type_=str, _default_="",
                        _help_="use decrypt_lib to load encrypted bmodel")
    _# yapf: enable_
    args = parser.parse_args()
    data = np.load(args.input)
    output = dict()
    _if_ args.model.endswith("final.mlir"):
        output = final_mlir_inference(data, args.model, args.dump_all_tensors)
    _elif_ args.model.endswith(".mlir"):
        output = mlir_inference(data, args.model, args.dump_all_tensors, args.debug, args.out_fixed,
                                args.cuda)
    _elif_ args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, args.dump_all_tensors)
    _elif_ args.model.endswith(".tflite"):
        output = tflite_inference(data, args.model, args.dump_all_tensors)
    _elif_ args.model.endswith(".prototxt") and args.weight.endswith(".caffemodel"):
        output = caffe_inference(data, args.model, args.weight, args.dump_all_tensors)
    _elif_ args.model.endswith(".pt") or args.model.endswith(".pth"):
        output = torch_inference(data, args.model, args.dump_all_tensors)
    _elif_ args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        output = model_inference(data,
                                 args.model,
                                 _out_fixed_=args.out_fixed,
                                 _decrypt_lib_=args.decrypt_lib)
    _else_:
        _raise_ RuntimeError("not support modle file:{}".format(args.model))
    print("\nSaving ...")
    _if_ output:
        np.savez(args.output, **output)
        print("\nResult saved to:{}".format(args.output))
```

- final_mlir_inference：针对 .final.mlir 文件的推理，调用 pyfinalmlir 库，支持 dump 所有张量。
- mlir_inference：针对 .mlir 文件的推理，支持 CPU/CUDA，支持 dump 所有张量。
- onnx_inference：针对 .onnx 文件的推理，支持 dump 所有中间张量（通过修改 ONNX graph 的 output）。
- tflite_inference：针对 .tflite 文件的推理，支持 dump 所有中间张量，支持 NCHW/NHWC 格式转换。
- caffe_inference：针对 Caffe 的 prototxt/caffemodel 文件推理，支持 dump 所有中间张量。
- torch_inference：针对 PyTorch 的 pt/pth 文件推理，支持 dump 所有中间张量。
- model_inference：针对 bmodel/cvimodel 文件推理，支持硬件仿真、文件锁、dump 所有张量等。

### 三、其他函数功能汇总

1. show_fake_cmd

功能：打印可复现的命令行调用示例，方便调试和复现推理过程。

原理：通过字符串格式化将参数拼接成完整命令行字符串，直接输出到控制台。

1. get_chip_from_model

功能：获取模型对应的芯片类型。

原理：调用外部命令 `model_tool --chip` 解析模型文件，通过 `os.popen` 执行命令并读取输出结果，适用于 bmodel、cvimodel 等需要绑定芯片类型的模型格式。

1. pack_bmodel_context_generator

功能：保存 bmodel 的输入和输出数据到文件。

原理：利用 Python 生成器（`yield`）机制，先保存输入数据后暂停执行，待推理完成后恢复执行并保存输出数据，实现推理前后的数据记录。

1. ChipLock

功能：硬件资源锁类，用于多进程 / 多线程环境下管理硬件资源（如仿真器），防止资源冲突。

原理：

1. model_inference

功能：BModel/CVIMODEL 的统一推理入口。

原理：

1. get_cmodel_so / link_custom_so / link_cmodel_so

功能：选择并链接正确的仿真库（.so 文件），确保仿真环境匹配芯片类型。

原理：

1. _model_inference

功能：BModel/CVIMODEL 的底层推理实现。

原理：

1. final_mlir_inference

功能：针对 `.final.mlir` 文件的推理。

原理：

1. mlir_inference

功能：针对 `.mlir` 文件的推理，支持 CPU 和 CUDA 环境。

原理：

1. _mlir_inference_by_cpu

功能：MLIR 模型在 CPU 环境下的底层推理实现。

原理：

1. _mlir_inference_by_cuda

功能：MLIR 模型在 CUDA 环境下的底层推理实现。

原理：

1. free_mlir_module

功能：释放全局 MLIR 模块资源。

原理：将全局 MLIR 模块变量设为 `None`，触发 Python 垃圾回收机制，避免长期运行时的内存泄漏。

1. onnx_inference

功能：ONNX 模型的推理实现。

原理：

1. caffe_inference

功能：Caffe 模型的推理实现。

原理：

1. tflite_inference

功能：TFLite 模型的推理实现。

原理：

1. torch_inference

功能：PyTorch 模型（`torch.jit` 导出的模型）的推理实现。

原理：

## 5.npz_tool.py

### 一、整体功能

这个脚本是一个多功能 npz 文件处理工具，用于对 .npz 文件（numpy 的多数组存储格式）进行各种操作，如对比、可视化、提取、合并、重命名、格式转换、统计等。它通过命令行参数选择不同的功能，适合模型开发、调试、量化、部署等场景下对中间/输出数据的批量处理。

### 二、主要结构与逻辑

1. 功能映射字典

```sql
npz_tool_func = {
    "compare": npz_compare,
    "print_diff": npz_print_diff,
    "visualize_diff": npz_visualize_diff,
    'dump':npz_dump,
    ...
}
```

- 用于根据命令行参数动态调用对应的处理函数。
- 每个 key 是命令行功能名，value 是实际执行的函数。

1. main 函数

```python
def main():
    args_list = sys.argv
    if len(args_list) < 2:
        funcs = npz_tool_func.keys()
        funcs_str = "["
        for idx, key in enumerate(npz_tool_func.keys()):
            funcs_str += key + ("|" if idx != len(funcs)-1 else '')
        funcs_str += "]"
        print(f"Usage: {args_list[0]} " + funcs_str + " ...")
        exit(-1)

    def NoneAndRaise(func):
        raise RuntimeError("No support {} Method".format(func))

    npz_tool_func.get(args_list[1], lambda x: NoneAndRaise(args_list[1]))(args_list[2:])
```

整体逻辑分为 3 步：

1. 参数数量检查：判断用户是否传入了至少 1 个功能名称（`len(args_list) < 2`）。若未传入，进入下一步提示用法；若已传入，直接执行对应功能。
2. 用法提示生成：当参数不足时，自动收集 `npz_tool_func` 字典中所有支持的功能名称，拼接成 `[功能1|功能2|...]` 格式的提示文本，告知用户工具的正确用法（如 `python script.py [功能名] ...`），然后退出。
3. 功能调用与错误处理：根据用户传入的功能名（`args_list[1]`），从 `npz_tool_func` 中查找对应的函数并调用（传入剩余参数 `args_list[2:]`）。若功能名不存在，通过 `NoneAndRaise` 函数抛出异常，提示 “不支持该功能”。

## 6.tpuc-opt.cpp

### 一、整体功能

这个文件实现了一个 TPU-MLIR 优化器命令行工具（类似于官方 MLIR 的 mlir-opt），用于对 MLIR（多层次中间表示）模块进行优化和转换。它是 TPU-MLIR 工具链中的核心组件之一，负责加载、注册所有自定义 Dialect 和 Pass，并根据命令行参数执行优化流程。

### 二、main 函数

#### 1）注册 Pass 和 Dialect

```python
tpu_mlir::registerAllPasses();
DialectRegistry registry;
tpu_mlir::registerAllDialects(registry);
```

- 注册所有自定义 Pass 和 Dialect，保证后续优化流程可用。

#### 2）处理 --deinit 选项

```go
bool hasDeinit = false;
for (int i = 1; i < argc; ++i) {
  if (std::string(argv[i]).find("--deinit") != std::string(argv[i]).npos) {
    hasDeinit = true;
    break;
  }
}
if (hasDeinit) {
  PluginPostPass = {"--mlir-print-debuginfo"};
} else {
  PluginPostPass = {"--deinit", "--mlir-print-debuginfo"};
}
```

- 检查命令行参数是否包含 --deinit。
- 如果有，则只加 --mlir-print-debuginfo，否则加 --deinit 和 --mlir-print-debuginfo。
- 这样可以灵活控制 pass 流程，避免重复 deinit。

#### 3）参数数量判断

```python
if (argc <= 2) {
  return asMainReturnCode(MlirOptMain(
      argc, argv, "TPU MLIR module optimizer driver\n", registry));
}
```

- 如果参数很少，直接调用 MLIR 的主入口，不做特殊处理。

#### 4）处理 --debug_cmd

```python
std::string debug_cmd = argv[argc - 1];
std::string substring = "--debug_cmd=";
if (debug_cmd.find(substring) != std::string::npos) {
  std::ofstream ofs;
  ofs.open("/tmp/debug_cmd", std::ios::out | std::ios::trunc);
  ofs << debug_cmd.substr(substring.size()) << std::endl;
  argc -= 1;
}
```

- 如果最后一个参数是 --debug_cmd=xxx，则将 xxx 写入 /tmp/debug_cmd 文件，并从参数列表中移除该参数。
- 便于调试和外部工具集成。

#### 5）构造新的参数列表

```python
int num_pre = sizeof(PluginPrePass) / sizeof(PluginPrePass[0]);
int num_post = PluginPostPass.size();
int new_argc = num_pre + argc + num_post;
char *new_argv[new_argc];
...
```

- 计算新参数数量（原始参数 + pre pass + post pass）。
- 构造新的参数数组 new_argv，将 pre pass、原始参数、post pass 按顺序插入。

##### 参数插入逻辑：

- 先插入原始参数（直到遇到第一个以 -- 开头的参数）。
- 插入 pre pass（如果原始参数中没有同名 pass）。
- 插入剩余原始参数（直到遇到 -o）。
- 插入 post pass。
- 插入剩余参数。

#### 6）调用 MLIR 主入口

```python
return asMainReturnCode(MlirOptMain(
    new_argc, new_argv, "TPU MLIR module optimizer driver\n", registry));
```

- 调用 MLIR 的主入口 MlirOptMain，传入新参数列表和注册表，执行优化流程。
