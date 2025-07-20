Writing by Jabari


以此视频教程为例https://www.bilibili.com/video/BV1JCZzYFEEJ/?share_source=copy_web&vd_source=4d9d633c1e01e9c9b929fe8311e2ad5b

github地址https://github.com/violetDelia/MLIR-Tutorial

mlir官方文档https://mlir.llvm.org/

llvm项目地址https://github.com/llvm/llvm-project

# 拉取项目地址

https://github.com/violetDelia/MLIR-Tutorial/blob/main/readme.md

## 构建命令：

1. 拉包：`git clone https://github.com/violetDelia/MLIR-Tutorial.git`
2. 拉取第三方库：`git submodule update --init --recursive`
3. build：cd `./MLIR-Tutorial && cmake . -G Ninja -B build`
4. 编译：cd build && ninja CH-2 [编译/测试命令]

​	需要用到cmake、ninja、gcc等工具，需要注意的是，gcc编译器的版本不能太低，不然会出现编译不通过的情况。

​	cmake下载地址https://cmake.org/download/

​	ninja下载地址https://github.com/ninja-build/ninja/releases/latest

​	gcc下载地址https://winlibs.com/

​	注意下载完成后一定要配置到系统环境变量中！

​	vscode中推荐使用clangd插件。

# 第一章.自定义方言

​	方言的定义：在 MLIR（Multi - Level Intermediate Representation）中，方言（Dialect）是一种用于描述特定领域或语言子集的结构。定义方言可以使 MLIR 适应不同的应用场景，如特定的编程语言、硬件描述、深度学习框架等。

​	通常使用TableGen语法进行方言的定义，包括方言名称、操作（ops）、类型（Types）等。

​	如开头链接中的项目为例。

​	在2-define_dialect\include\Dialect\NorthStar\IR\NorthStarDialect.td文件中的代码。见代码注释。

​	需要注意的是在 MLIR 的 TableGen 语法中，`let` 关键字用于配置和自定义类、方言、操作等实体的属性。它的作用是覆盖或设置基类中定义的默认属性值

```TableGen
//定义自己的方言，继承自Dialect，这里的Dialect是MLIR的基础方言定义。在MLIR-Tutorial\third_party\llvm-project\mlir\include\mlir\IR\DialectBase.td
//定义的语法是let，用于覆盖基类中的属性和方法。
def NorthStar_Dialect : Dialect{
  // 方言的名字
  let name = "north_star";

  // 方言的概述
  let summary = "summary of NorthStar Dialect";

  // 方言的详细描述
  let description = "description of NorthStar Dialect";

  // 方言的依赖
  // 方言依赖（Dependent Dialects） 指的是一个方言在功能上依赖于其他方言的特性。当你定义一个方言时，通过指定依赖关系，可以确保该方言能够正确使用其他方言定义的类型、操作或属性。
  // 这里依赖的是TensorDialect，TensorDialect是MLIR的一个内置方言，提供了对张量类型和操作的支持。
  let dependentDialects = ["::mlir::tensor::TensorDialect"];

  // 用于生成比较标准的属性管理的代码 [4-7]
  // discardableAttrs 的作用
  // 这个选项用于指定哪些属性在操作（Operation）被转换或优化时可以被安全地丢弃。在 MLIR 的编译流程中，某些属性可能只在特定阶段有用，后续阶段可以忽略它们。
  // (ins) 的含义
  // (ins) 是一个 TableGen 语法，表示 "无参数"。在这里，它意味着没有定义可丢弃的属性。也就是说，该方言中的所有属性都被视为不可丢弃，在转换过程中会被保留。
  let discardableAttrs = (ins);

  // 生成代码在C++代码中的命名空间
  // 一般为::mlir::dialect_name
  let cppNamespace = "::mlir::north_star";

  // 额外的声明.
  // 在 MLIR 的 TableGen 方言定义中，extraClassDeclaration 选项允许你向自动生成的 C++ 类添加额外的声明。这是一种扩展生成代码的机制，用于添加 TableGen 本身不直接支持的自定义功能。
  // 这段代码的作用是：
  // 1.向生成的 NorthStarDialect C++ 类添加一个静态成员函数 sayHello()。
  // 2.这个函数不会由 TableGen 自动实现，需要你手动在 C++ 代码中提供实现。
  let extraClassDeclaration = [{
    static void sayHello();
  }];

  // 规范化的声明. [14]
  // 0表示没有规范化的声明，1表示有规范化的声明。
  let hasConstantMaterializer = 0;

  // 是否生成默认的析构函数
  // 0表示生成，1表示不生成。
  let hasNonDefaultDestructor = 1;

  // 操作数的属性检验 [7]
  let hasOperationAttrVerify = 0;

  // RegionArg的属性检验 [7]
  let hasRegionArgAttrVerify = 0;

  // RegionResult的属性检验 [7]
  let hasRegionResultAttrVerify = 0;

  // [6]
  let hasOperationInterfaceFallback = 0;

  // 使用MLIR默认的属性解析输出.
  // 0为不使用，1为使用
  let useDefaultAttributePrinterParser = 0;

  // 使用MLIR默认的类型解析输出.
  let useDefaultTypePrinterParser = 0;

  // 是否有规范化patten[14].
  let hasCanonicalizer = 0;

  // 是否是可扩展的方言.
  let isExtensible = 0;

  // Whether inherent Attributes defined in ODS will be stored as Properties.
  let usePropertiesForAttributes = 1;

}
```

​	定义完方言后使用cmake代码将TableGen文件编译为C++文件。代码如下，地址在2-define_dialect\include\Dialect\NorthStar\IR\CMakeLists.txt

```cmake
set(LLVM_TARGET_DEFINITIONS NorthStarDialect.td)		#设置TableGen源文件，LLVM_TARGET_DEFINITIONS 是 LLVM/MLIR 的 CMake 变量，用于指定 TableGen 的输入文件，NorthStarDialect.td是定义的方言文件
# 生成NorthStar Dialect 的声明
#	mlir_tablegen 是 MLIR 提供的 CMake 宏，用于调用 TableGen 工具。
#	--gen-dialect-decls 是 TableGen 的参数，指示生成 C++ 头文件声明（如类定义、方法签名）。
#	-dialect=north_star 指定要处理的方言名称。
#	生成的 NorthStarDialect.h.inc 会被包含在最终的 C++ 头文件中。
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
#	--gen-dialect-defs 指示生成 C++ 方法的实现代码。
#	生成的 NorthStarDialect.cpp.inc 会被包含在最终的 C++ 实现文件中。
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)

# 将生成的命令们定义为为target
#	add_public_tablegen_target 是 LLVM 提供的宏，用于创建一个公共的 TableGen 构建目标。
#	这个目标负责将 TableGen 描述文件（.td）编译为 C++ 代码（.h.inc和.cpp.inc），并确保这些生成的文件在项目的其他部分被正确依赖和使用。
add_public_tablegen_target(NorthStarDialectIncGen${ch_num})
```

​	上面代码中的.inc作用：

​	在 MLIR 项目中，`.inc` 是一种**约定俗成的文件扩展名**，通常表示 “包含文件”（**Include File**）。这些文件不是独立的完整源文件，而是用于**被其他 C++ 文件包含**的代码片段，类似于头文件（`.h`），但有特定的用途：

**为什么使用 `.inc` 文件？**

1. **代码生成的模块化**：
   - MLIR 使用 TableGen 自动生成大量代码（如操作定义、类型系统）。这些生成的代码被分割成`.inc`文件，便于管理和维护。
   - 例如，`NorthStarDialect.h.inc` 包含由 TableGen 生成的方言类声明，而`NorthStarDialect.cpp.inc` 包含对应的实现。
2. **避免重复定义**：
   - `.inc` 文件通常不包含头文件保护（如`#ifndef DIALECT_NORTH_STAR`），因为它们会被包含在其他已经有保护的文件中。
3. **与项目结构解耦**：
   - 生成的`.inc`文件通常放在构建目录（如`build/include/`），而不是源码目录，保持源码树的整洁。

​	上述代码实现后，在终端使用命令cd build/进入build目录，然后输入命令ninja NorthStarDialectIncGen2编译文件，编译后可以得到C++代码的头文件和源文件，头文件路径为build\2-define_dialect\include\Dialect\NorthStar\IR\NorthStarDialect.h.inc	源文件路径为build\2-define_dialect\include\Dialect\NorthStar\IR\NorthStarDialect.cpp.inc

​	头文件代码如下,头文件（通常是`.h`或`.h.inc`）的核心作用是**声明函数、类和类型**，为 C++ 代码提供接口定义。

```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NorthStarDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {			//这是 MLIR 框架的顶级命名空间，所有 MLIR 核心库和官方方言都位于此命名空间下。
namespace north_star {		//这是你自定义方言的专属命名空间，用于隔离该方言内的所有类、函数和类型。

class NorthStarDialect : public ::mlir::Dialect {
  explicit NorthStarDialect(::mlir::MLIRContext *context);		//构造函数声明

  void initialize();		//初始化方法声明
  friend class ::mlir::MLIRContext;
public:
  ~NorthStarDialect() override;		//析构函数，标记为override确保正确覆盖基类虚函数
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("north_star");
  }

    static void sayHello();		//NorthStarDialect.td文件里的自定义函数
  };
} // namespace north_star
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::north_star::NorthStarDialect)		//这是 MLIR 为方言类声明唯一类型 ID 的宏，用于运行时类型识别和注册。

```

​	头文件中定义的函数在源文件中实现，源文件代码如下：

```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NorthStarDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::north_star::NorthStarDialect)
namespace mlir {
namespace north_star {

//类外实现构造函数
//声明函数，表明这是 NorthStarDialect 类的构造函数。参数类型为 MLIR 上下文的指针::mlir::MLIRContext *context
//调用基类 mlir::Dialect 的构造函数进行初始化。	getDialectNamespace()：返回方言的命名空间字符串（在 .td 文件中定义，如 "north_star"）。		context：传递 MLIR 上下文指针。	::mlir::TypeID::get<NorthStarDialect>()：获取方言的唯一类型 ID，用于运行时类型识别。
NorthStarDialect::NorthStarDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NorthStarDialect>())
    
     {
  getContext()->loadDialect<::mlir::tensor::TensorDialect>();		//加载依赖方言、这个依赖是在NorthStarDialect.td文件中定义的依赖
  initialize();		//调用初始化方法
}
} // namespace north_star
} // namespace mlir

```

​	头文件和源文件编译完成后，我们要去实现。用于实现的文件路径为：

​	头文件：2-define_dialect\include\Dialect\NorthStar\IR\NorthStarDialect.h

​	源文件：2-define_dialect\src\Dialect\NorthStar\IR\NorthStarDialect.cpp

​	头文件代码如下

```cpp
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef DIALECT_NORTH_STAR_H
#define DIALECT_NORTH_STAR_H
#include "mlir/Dialect/Tensor/IR/Tensor.h"		//引入 MLIR 的 Tensor 方言支持，对应.td文件中的dependentDialects。
#include "mlir/IR/MLIRContext.h"			//引入 MLIR 上下文相关的定义，用于方言初始化。
#include "Dialect/NorthStar/IR/NorthStarDialect.h.inc"		//通过#include引入 TableGen 生成的.inc文件（而非直接编写 C++ 代码）。

#endif  // DIALECT_NORTH_STAR_H
```

​	源文件代码如下

```cpp
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include "Dialect/NorthStar/IR/NorthStarDialect.h"		//引入之前定义的头文件（包含方言类声明）

#include "llvm/Support/raw_ostream.h"				   //引入 LLVM 的输出流支持，用于日志输出
#define FIX
#include "Dialect/NorthStar/IR/NorthStarDialect.cpp.inc"
#undef FIX

namespace mlir::north_star {
// 实现方言的初始化方法
void NorthStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";		//getDialectNamespace()返回方言的命名空间字符串
}

// 实现方言的析构函数
NorthStarDialect::~NorthStarDialect() {
  llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}

// 实现在extraClassDeclaration 声明当中生命的方法。即自定义方法的实现
void NorthStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}

}  // namespace mlir::north_star

```

​	下面这段 CMake 代码是 MLIR 项目中用于构建方言库的标准配置，路径为2-define_dialect\src\Dialect\NorthStar\IR\CMakeLists.txt

```cmake
#add_mlir_dialect_library：MLIR 提供的 CMake 宏，用于创建方言库
#MLIRNorthStarDialect${ch_num}：目标库名称，${ch_num}是变量（如版本号）
add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp		#源文件，包含方言实现代码

    DEPENDS		#依赖管理,指定构建依赖关系
    NorthStarDialectIncGen${ch_num}		#TableGen 生成目标（来自之前的add_public_tablegen_target）

    LINK_LIBS PUBLIC		#指定需要链接的公共库
    MLIRIR		#MLIR 核心库，提供基础类型和操作
    MLIRTensorDialect		#Tensor 方言库，对应.td中的dependentDialects
)


```

​	在main函数中实现

```cpp
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include "Dialect/NorthStar/IR/NorthStarDialect.h"		//自定义方言的主头文件
#include "mlir/IR/DialectRegistry.h"				   //MLIR 的方言注册器，用于管理可用方言
#include "mlir/IR/MLIRContext.h"					  //MLIR 的核心上下文，所有操作都在上下文中执行

void CH2() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();		//getOrLoadDialect<...>()模板方法，用于获取指定方言的实例。如果方言尚未加载，则自动加载并注册。

  // 调用方言中的方法
  dialect->sayHello();
}

int main() { CH2(); }
```

​	

```cmake
set(ch_num 2)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
add_executable(CH-${ch_num} "main.cpp")
target_link_libraries(CH-${ch_num} MLIRNorthStarDialect${ch_num})		#将CH-2可执行文件与MLIRNorthStarDialect2库链接起来

```

​	然后在终端输入命令ninja CH-2进行编译，编译后在构建目录下会生成 `CH-2` 可执行文件，路径为build\2-define_dialect\CH-2.exe，运行该文件后，终端输出

initializing north_star
Hello in north_star
destroying north_star

​	这样就完成了方言的注册、加载以及初始化。

# 第二章.内建type及自定义type

## 	2.1**内建Type**	

​	MLIR提供了一系列内建类型，这些是 MLIR 的核心组成部分，用于描述张量、标量、内存、函数等抽象结构。文件路径为third_party\llvm-project\mlir\include\mlir\IR\BuiltinTypes.td，下面列举部分内建类型。一些常用内建类型可以看链接视频。

​	1.标量类型

| 类型             | 描述                                                         |
| ---------------- | ------------------------------------------------------------ |
| iN               | 整型（例如i32，i64，i1），其中N是位宽                        |
| f16,f32,f64,bf16 | 浮点类型（半精度、单精度、双精度、bfloat）                   |
| index            | 表示索引类型，用于数组索引等，取决于系统，32位系统则是32位无符号整数，64位系统则是64位无符号整数。 |
| complex<T>       | 复数类型，例如complex<f64>                                   |

​	2.容器类型

​	Tensor：用于表示不可变的张量数据

| tensor<2x3xf32> | 2x3的float32张量 |
| --------------- | ---------------- |
| tensor<?x?xi32> | 动态张量的类型   |

​	MemRef：表示内存引用，常用于buffer

| memref<4x4xi8> | 4x4的i8 buffer         |
| -------------- | ---------------------- |
| memref<?xf32>  | 动态大小的float buffer |

​	3.结构类型

| tuple<i32,f32,tensor<2xf64>> | 元组类型，表示多个值组合                                     |
| ---------------------------- | ------------------------------------------------------------ |
| (i32,f32) -> (f32)           | 函数类型，表示输入输出的类型，接受i32和f32参数，并返回一个f32类型的值 |

​	4.其他类型

| NoneType   | 表示无值，如Python的None和C++的void           |
| ---------- | --------------------------------------------- |
| OpaqueType | 用于 dialect 定义自己的特殊类型，外部无法识别 |

## 	2.2自定义Type

​	首先要编写自定义type的tablegen文件，可以先仿造llvm项目里的BuiltinTypes.td文件中的代码来创建一个基类，路径为third_party\llvm-project\mlir\include\mlir\IR\BuiltinTypes.td

```tablegen
class Builtin_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">    
    : TypeDef<Builtin_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let typeName = "builtin." # typeMnemonic;
}   //类型名称 name、类型助记符 typeMnemonic、类型特性列表 traits（默认为空列表），以及 C++ 基类名 baseCppClass（默认为 ::mlir::Type）。
```

​	然后修改为自己要定义的Type的信息

```tablegen
class NorthStar_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">    
    : TypeDef<Builtin_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  //tablegen语法中#有连接字符串的作用
  let typeName = dialect.name # "." # typeMnemonic;
}
```

​	mlir内建的tensor无法表示其内存层次的信息，现在定义一个tensor类型，让其可以表示内存层次的信息。

```Tablegen
def NorthStar_TensorType : 
  //定义NorthStar_TensorType，继承自基类NorthStar_Type，其参数为：name为NSTensor，typeMnemonic为ns_tensor，list为空列表
NorthStar_Type<"NSTensor","ns_tensor",[]>{
  // 概述
  let summary = " the summary of north-star tensor type";

  // 详细描述
  let description = "description of north-star tensor type";

  // 参数
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,		//张量的形状
    "Type":$elementType,		//张量元素的类型
    "int64_t":$device_id		//内存设备的信息
  );

  // 是否生成StorageClass, 无特殊情况，建议设为ture
  // 在 MLIR 的 TableGen 定义中，let genStorageClass = 1; 这一行的作用是指示 MLIR 自动生成与类型相关的存储类（Storage Class）。
  // 作用为：存储类型的参数（如张量的形状、元素类型）、实现类型的唯一性检查（通过 TypeStorage 基类）、提供类型的属性访问方法
  // 例如，对于 NorthStar_TensorType，其存储类会保存 shape、elementType 和 device_id 三个参数。
  let genStorageClass = 1;
  
  // 在 MLIR 的 TableGen 定义中，let hasStorageCustomConstructor = 0; 这一设置控制着存储类（Storage Class）的构造方式。
  // 表示是否生成StorageClass默认的构造函数，0表示生成，1表示不生成，手动构造
  let hasStorageCustomConstructor = 0;

  // 额外的builder 声明
  // 只有两个参数，device默认为0 
  let builders = [
    TypeBuilder<(ins 
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType),[{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;		//是否使用mlir默认的Format函数序列化Type，0为使用
     //assemblyFormat 在 MLIR TableGen 中的作用是：自动生成操作或类型的文本格式（assembly）的 parse() 和 print() 函数，你就不需要手动写 C++ 代码来	    //解析或打印 IR 了。
  //let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";	

  // 不跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

  // extraClassDeclaration 用来插入额外的 C++ 代码声明，扩展类型类。这里注释了多条 using 声明和一个 clone 函数,这在第五章时会用到
  let extraClassDeclaration = [{
    // using TensorType::clone;
    // using ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
    // using ShapedType::Trait<NSTensorType>::getRank;
    // using ShapedType::Trait<NSTensorType>::getNumElements;
    // using ShapedType::Trait<NSTensorType>::isDynamicDim;
    // using ShapedType::Trait<NSTensorType>::hasStaticShape;
    // using ShapedType::Trait<NSTensorType>::getNumDynamicDims;
    // using ShapedType::Trait<NSTensorType>::getDimSize;
    // using ShapedType::Trait<NSTensorType>::getDynamicDimIndex;
    // NSTensorType clone(::mlir::Type elementType) {
    //   return ::llvm::cast<NSTensorType>(cloneWith(getShape(), elementType));
    // }
  }];
}
```

​	定义完成后，将代码生成。路径为3-define_type\include\Dialect\NorthStar\IR\CMakeLists.txt

```cmake
set(LLVM_TARGET_DEFINITIONS NorthStarTypes.td)		#设置 LLVM_TARGET_DEFINITIONS，告诉 mlir_tablegen 工具接下来要处理哪个 TableGen 文件。
# 生成NorthStar Dialect 的声明
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)
# 生成NorthStar Type 的声明
mlir_tablegen(NorthStarTypes.h.inc -gen-typedef-decls -dialect=north_star)
# 生成NorthStar Type 的实现
mlir_tablegen(NorthStarTypes.cpp.inc -gen-typedef-defs -dialect=north_star)
# 将生成的命令们定义为为target
add_public_tablegen_target(NorthStarDialectIncGen${ch_num})

```

​	生成的头文件为，路径为build\3-define_type\include\Dialect\NorthStar\IR\NorthStarTypes.h.inc

```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace north_star {
class NSTensorType;
namespace detail {
struct NSTensorTypeStorage;		//类型数据的实际存储结构,是由tablegen文件中 let genStorageClass = 1; 自动生成的！
} // namespace detail
    
 //class NSTensorType
 //  : public ::mlir::Type::TypeBase<
 //    NSTensorType,          // 当前类型类名
 //    ::mlir::Type,          // 基类，所有 Type 的基类
 //    detail::NSTensorTypeStorage>  // 存储结构（下一节解释）
 // NSTensorType 是一个 MLIR 类型类（TypeBase）,继承了 mlir::Type 的所有功能，类型的参数数据（shape, elementType, device_id）由      NSTensorTypeStorage 存储
class NSTensorType : public ::mlir::Type::TypeBase<NSTensorType, ::mlir::Type, detail::NSTensorTypeStorage> {
public:
  using Base::Base;
  // using TensorType::clone;
  // using ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
  // using ShapedType::Trait<NSTensorType>::getRank;
  // using ShapedType::Trait<NSTensorType>::getNumElements;
  // using ShapedType::Trait<NSTensorType>::isDynamicDim;
  // using ShapedType::Trait<NSTensorType>::hasStaticShape;
  // using ShapedType::Trait<NSTensorType>::getNumDynamicDims;
  // using ShapedType::Trait<NSTensorType>::getDimSize;
  // using ShapedType::Trait<NSTensorType>::getDynamicDimIndex;
  // NSTensorType clone(::mlir::Type elementType) {
  //   return ::llvm::cast<NSTensorType>(cloneWith(getShape(), elementType));
  // }
  static constexpr ::llvm::StringLiteral name = "north_star.ns_tensor";		//为类型定义了一个编译期的字符串常量，用来唯一标识这个类型。
  static constexpr ::llvm::StringLiteral dialectName = "north_star";		//为这个类型声明所属的 Dialect（方言）名称
  using Base::getChecked;
  //默认builder的get方法
  static NSTensorType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id);
  //默认builder的getcheck方法
  static NSTensorType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id);
  //额外builder的get方法
  static NSTensorType get(::mlir::MLIRContext *context, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType);
  //额外builder的getcheck方法
  static NSTensorType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType);
  //参数校验函数，由tablegen文件中  let genVerifyDecl = 1; 生成
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ns_tensor"};
  }

  //parse() 和 print()	需要在源文件中自己实现
  //负责将类型和 IR 文本互相转换，由tablegen文件中  let genVerifyDecl = 1; 生成
  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
  int64_t getDeviceId() const;
};
} // namespace north_star
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::north_star::NSTensorType)

#endif  // GET_TYPEDEF_CLASSES


```

​	下面代码是由 TableGen 根据类型定义自动生成的，负责定义和实现自定义的 MLIR 类型 `NSTensorType` 及其相关辅助函数。路径为build\3-define_type\include\Dialect\NorthStar\IR\NorthStarTypes.cpp.inc

```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::north_star::NSTensorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

    
//generatedTypeParser,这是一个类型解析的辅助函数。它用 KeywordSwitch 根据解析到的类型名（mnemonic）分发调用对应类型的 parse() 函数。如果匹配到 NSTensorType 的 mnemonic（比如 "ns_tensor"），就调用 NSTensorType::parse()。如果找不到对应类型，返回 nullopt 表示解析失败，由调用方处理错误。
static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::north_star::NSTensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::north_star::NSTensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

//generatedTypePrinter,这是类型打印的辅助函数。使用 TypeSwitch 对传入的 Type 进行动态类型匹配。如果是 NSTensorType，先打印它的 mnemonic（比如 "ns_tensor"），然后调用该类型实例的 print() 方法打印参数内容。不支持的类型返回失败。
static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::north_star::NSTensorType>([&](auto t) {
      printer << ::mlir::north_star::NSTensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace north_star {
namespace detail {
////NSTensorTypeStorage 结构体,NSTensorType 类型的 存储类，继承自 mlir::TypeStorage。用于持有类型的“参数”数据：shape、elementType、device_id。
struct NSTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, int64_t>;
  NSTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) : shape(std::move(shape)), elementType(std::move(elementType)), device_id(std::move(device_id)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, device_id);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (device_id == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static NSTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto device_id = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<NSTensorTypeStorage>()) NSTensorTypeStorage(std::move(shape), std::move(elementType), std::move(device_id));
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  int64_t device_id;
};
} // namespace detail
//  //NSTensorType 的静态工厂方法实现,内部调用 Base::get()，由 MLIR 的 TypeBase 模板类实现，利用 NSTensorTypeStorage 进行唯一性缓存和返回类型实例。同理，getChecked 会先调用你实现的 verify 函数做合法性校验。
NSTensorType NSTensorType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(device_id));
}

NSTensorType NSTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
  return Base::getChecked(emitError, context, shape, elementType, device_id);
}

NSTensorType NSTensorType::get(::mlir::MLIRContext *context, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) {
  return Base::get(elementType.getContext(), shape, elementType, 0);
}

NSTensorType NSTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), shape, elementType, 0);
}

//访问类型参数的接口实现
::llvm::ArrayRef<int64_t> NSTensorType::getShape() const {
  return getImpl()->shape;
}

Type NSTensorType::getElementType() const {
  return getImpl()->elementType;
}

int64_t NSTensorType::getDeviceId() const {
  return getImpl()->device_id;
}

} // namespace north_star
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::north_star::NSTensorType)
namespace mlir {
namespace north_star {

/// Parse a type registered to this dialect.
::mlir::Type NorthStarDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void NorthStarDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace north_star
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES


```

​	下面代码是自定义的 `NSTensorType` 类型的核心 C++ 实现，配合 TableGen 自动生成的声明和存储类，完成 MLIR 类型的解析、打印和验证功能。路径为3-define_type\src\Dialect\NorthStar\IR\NorthStarTypes.cpp

```cpp
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include "Dialect/NorthStar/IR/NorthStarTypes.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#define FIX
#define GET_TYPEDEF_CLASSES
#include "Dialect/NorthStar/IR/NorthStarTypes.cpp.inc"

namespace mlir::north_star {

//这是 Dialect 中注册自定义类型的函数。
void NorthStarDialect::registerType() {
  llvm::outs() << "register " << getDialectNamespace() << "  Type\n";
  addTypes<		//addTypes 是 MLIR 方言接口，用来批量注册类型。
#define GET_TYPEDEF_LIST
#include "Dialect/NorthStar/IR/NorthStarTypes.cpp.inc"
      >();
}

//这是类型参数校验函数，在调用 getChecked 时会自动被调用。
::llvm::LogicalResult NSTensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
  if (device_id < 0) {		//device_id 不能小于0。
    return emitError() << " Invalid device id";
  }
  if (!elementType.isIntOrFloat()) {		//elementType 必须是整型或浮点型。
    return emitError() << " Invalid element type ";		//不符合条件时通过 emitError() 输出错误，返回失败。
  }
  return llvm::success();
}

//这是类型文本解析函数。
Type NSTensorType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();		//先读入符号 <

  SmallVector<int64_t, 4> dimensions;	//定义一个容器 dimensions，用来存储解析出来的维度信息。使用 SmallVector 是 LLVM/MLIR 的高效动态数组模板。
    
  //调用 MLIR 解析器的 parseDimensionList，从文本读取维度列表。
  //参数说明：dimensions：解析结果存入此数组。allowDynamic=true：允许维度为动态大小（用 ? 或 -1 表示）。withTrailingX=true：维度之间用 x 分隔，如 3x4x?x.
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return Type();	//解析失败时返回空类型 Type()。
  // Parse the element type.
  auto typeLoc = parser.getCurrentLocation();
  Type elementType;		//定义一个变量，准备存储解析出来的元素类型。
  if (parser.parseType(elementType)) return Type();		//解析一个 MLIR 类型文本，存入 elementType。失败时返回空类型 Type()。
  // Check that array is formed from allowed types.
  if (parser.parseComma()) return Type();		//期望读取逗号 , 分隔符。失败返回空类型。
  int device_id = 0;		//  int device_id = 0;
  if (parser.parseInteger(device_id))		//先解析一个整数，赋值给 device_id。
    if (parser.parseGreater()) return Type();		//如果解析整数失败，接着尝试解析结束符 >，失败就返回空类型。
  //调用 MLIR 提供的 getChecked 工厂方法构造类型实例。传入当前上下文、刚解析出来的参数：dimensions、elementType、device_id。getChecked 内部会调用你实现的 verify() 函数，确保参数合法。成功返回类型实例，失败返回空类型。
  return parser.getChecked<NSTensorType>(parser.getContext(), dimensions,
                                         elementType, device_id);
}

//这是类型打印函数。
void NSTensorType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    if (dim < 0) {
      printer << "?" << 'x';
    } else {
      printer << dim << 'x';
    }
  }
  printer.printType(getElementType());
  printer << ",";
  printer << getDeviceId();
  printer << ">";
}
}  // namespace mlir::north_star

#undef FIX

```

​	main文件，路径为3-define_type\main.cpp

```cpp
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include <cstddef>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

//测试方言能否正常注册
void CH2() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  // 调用方言中的方法
  dialect->sayHello();
}

//内建类型
void typeBrief() {
  // 文件定义：llvm-project/mlir/include/mlir/IR/BuiltinTypes.td
  auto context = new mlir::MLIRContext;

  // 浮点数，每种位宽和标准定义一个
  auto f32 = mlir::Float32Type::get(context);
  llvm::outs() << "F32类型 :\t";
  f32.dump();

  auto bf16 = mlir::BFloat16Type::get(context);
  llvm::outs() << "BF16类型 :\t";
  bf16.dump();

  // Index 类型，机器相关的整数类型
  auto index = mlir::IndexType::get(context);
  llvm::outs() << "Index 类型 :\t";
  index.dump();

  // 整数类型, 参数: 位宽&&有无符号
  auto i32 = mlir::IntegerType::get(context, 32);
  llvm::outs() << "I32 类型 :\t";
  i32.dump();
  auto ui16 = mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
  llvm::outs() << "UI16 类型 :\t";
  ui16.dump();

  // 张量类型,表示的是数据，不会有内存的布局信息。
  auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 张量类型 :\t";
  static_tensor.dump();
  // 动态张量
  auto dynamic_tensor =
      mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
  llvm::outs() << "动态F32 张量类型 :\t";
  dynamic_tensor.dump();

  // Memref类型：表示内存
  auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 内存类型 :\t";
  basic_memref.dump();
  // 带有布局信息的内存

  auto stride_layout_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32, mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}));
  llvm::outs() << "连续附带布局信息的 F32 内存类型 :\t";
  stride_layout_memref.dump();
  // 使用affine 表示布局信息的内存
  auto affine_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32,
      mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}).getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的 F32 内存类型 :\t";
  affine_memref.dump();
  // 动态连续附带 affine 布局信息的内存
  auto dynamic_affine_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
  dynamic_affine_memref.dump();
  // 具有内存层级信息的内存
  auto L1_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            1);
  llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
  L1_memref.dump();
  // gpu 私有内存层级的内存
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  auto gpu_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            mlir::gpu::AddressSpaceAttr::get(
                                context, mlir::gpu::AddressSpace::Private));
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型 :\t";
  gpu_memref.dump();

  // 向量类型,定长的一段内存
  auto vector_type = mlir::VectorType::get(3, f32);
  llvm::outs() << "F32 1D向量类型 :\t";
  vector_type.dump();

  auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
  llvm::outs() << "F32 2D向量类型 :\t";
  vector_2D_type.dump();
  delete context;
}

//注册并加载自定义方言。
void CH3() {
  typeBrief();
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  // 调用方言中的方法
  dialect->sayHello();
  // 静态 NSTensor
  mlir::north_star::NSTensorType ns_tensor =
      mlir::north_star::NSTensorType::get(&context, {1, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "North Star Tensor 类型 :\t";
  ns_tensor.dump();
  // 动态 NSTensor
  mlir::north_star::NSTensorType dy_ns_tensor =
      mlir::north_star::NSTensorType::get(&context,
                                          {mlir::ShapedType::kDynamic, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "动态 North Star Tensor 类型 :\t";
  dy_ns_tensor.dump();
}

int main() { CH3(); }
```

# 第三章.内建Attribute及自定义Attribute

## 3.1内建Attribute

​	在 **MLIR**（Multi-Level Intermediate Representation）中，**Attribute（属性）** 是一种 **不可变（immutable）** 的、用来描述附加信息的数据结构。简单的说，在 MLIR 中，**Attribute 就是“不可变的常量数据”**，比如数字、字符串、布尔值、数组、名字等等，用来给操作（operation）添加信息。

​	例如

```tablegen
%a = arith.constant 42 : i32	//这条语句表示给变量 %a 赋值一个常量 42。42 是一个 Attribute，类型是 IntegerAttr，意思是整数属性
```

```tablegen
%b = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>	//这里的 [1.0, 2.0] 是一个数组型属性。它表示一个固定的数组常量
```

​	MLIR 提供了一系列 内建属性 类型，文件路径为third_party\llvm-project\mlir\include\mlir\IR\BuiltinAttributes.td

​	一些常用的attribute如下

| 名称                 | C++ 类型                   | 示例                           | 说明                               |
| -------------------- | -------------------------- | ------------------------------ | ---------------------------------- |
| `BoolAttr`           | `mlir::BoolAttr`           | `true`                         | 布尔值属性                         |
| `IntegerAttr`        | `mlir::IntegerAttr`        | `42 : i32`                     | 整数属性，携带类型信息             |
| `FloatAttr`          | `mlir::FloatAttr`          | `3.14 : f32`                   | 浮点属性                           |
| `StringAttr`         | `mlir::StringAttr`         | `"hello"`                      | 字符串属性                         |
| `ArrayAttr`          | `mlir::ArrayAttr`          | `[42, "hello"]`                | 属性列表                           |
| `DenseElementsAttr`  | `mlir::DenseElementsAttr`  | `[1, 2, 3] : tensor<3xi32>`    | 稠密张量常量                       |
| `SparseElementsAttr` | `mlir::SparseElementsAttr` | -                              | 稀疏张量常量                       |
| `SymbolRefAttr`      | `mlir::SymbolRefAttr`      | `@func`                        | 引用另一个符号（函数、变量）       |
| `TypeAttr`           | `mlir::TypeAttr`           | `i32`                          | 用来存储类型                       |
| `UnitAttr`           | `mlir::UnitAttr`           | `unit`                         | 空属性，用于表示 flag              |
| `DictionaryAttr`     | `mlir::DictionaryAttr`     | `{name = "value"}`             | 键值对结构                         |
| `OpaqueAttr`         | `mlir::OpaqueAttr`         | -                              | 不可解析但可转发的属性             |
| `AffineMapAttr`      | `mlir::AffineMapAttr`      | `affine_map<(d0) -> (d0 * 2)>` | Affine Map 属性                    |
| `AffineExprAttr`     | `mlir::AffineExprAttr`     | `affine<d0 + 2>`               | Affine 表达式属性                  |
| `ElementsAttr`       | `mlir::ElementsAttr`       | 父类                           | 所有 `Dense/Sparse` 元素属性的基类 |
| `FlatSymbolRefAttr`  | `mlir::FlatSymbolRefAttr`  | `@func`                        | 一种简化版本的 SymbolRefAttr       |

​	内建属性的基类：代码定义了这段定义了一个 TableGen 类模板，继承类AttrDef，名字叫 `Builtin_Attr`，它有4个参数。

```tablegen
// Base class for Builtin dialect attributes.
class Builtin_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Builtin_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let attrName = "builtin." # attrMnemonic;
}
```

| 参数名         | 类型          | 说明                                                  |
| -------------- | ------------- | ----------------------------------------------------- |
| `name`         | `string`      | 这个属性在 C++ 中的类名，比如 `"BoolAttr"`            |
| `attrMnemonic` | `string`      | 用于 `.mlir` IR 文本语法中显示的关键字，比如 `"bool"` |
| `traits`       | `list<Trait>` | 该属性支持的 trait（如接口或校验逻辑）默认是空列表    |
| `baseCppClass` | `string`      | 指定该属性的 C++ 基类，默认是 `mlir::Attribute`       |

## 3.2自定义Attribute

​	编写tablegen的文件，路径为4-define_attribute\include\Dialect\NorthStar\IR\NorthStarAttrs.td

```tablegen
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef DIALECT_NORTH_STAR_ATTRS_TD
#define DIALECT_NORTH_STAR_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect/NorthStar/IR/NorthStarEunms.td"


class NorthStar_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<NorthStar_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = dialect.name # "." # attrMnemonic;
  let genStorageClass = 1;		//自动生成 Storage 类
  let hasStorageCustomConstructor = 0;		//不自定义 Storage 构造函数（用默认的）
  let skipDefaultBuilders = 0;	 //生成默认的builders
  let genVerifyDecl = 0;		//不生成 verify() 方法声明
}

//自定义 Attribute 实体
def NorthStar_DataParallelism: NorthStar_Attr<"DataParallelism", "DP", []>{
  let parameters = (ins "int64_t":$DP_nums);
  let assemblyFormat = [{
    `<`
      `DP` `=` $DP_nums
    `>`
  }];
}

#endif //DIALECT_NORTH_STAR_ATTRS_TD
```

​	然后通过cmake文件让 MLIR 的 TableGen 工具自动生成 NorthStar 方言相关的 C++ 文件

```cmake
set(LLVM_TARGET_DEFINITIONS NorthStarAttrs.td)
# 生成NorthStar Dialect 的声明
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)
# 生成NorthStar Type 的声明
mlir_tablegen(NorthStarTypes.h.inc -gen-typedef-decls -dialect=north_star)
# 生成NorthStar Type 的实现
mlir_tablegen(NorthStarTypes.cpp.inc -gen-typedef-defs -dialect=north_star)
# 生成NorthStar Attr 的声明
mlir_tablegen(NorthStarAttrs.h.inc -gen-attrdef-decls -dialect=north_star)
# 生成NorthStar Attr 的实现
mlir_tablegen(NorthStarAttrs.cpp.inc -gen-attrdef-defs -dialect=north_star)
# 将生成的命令们定义为为target
add_public_tablegen_target(NorthStarDialectIncGen${ch_num})
```

​	生成后的声明和实现分别在build\4-define_attribute\include\Dialect\NorthStar\IR\NorthStarAttrs.h.inc和build\4-define_attribute\include\Dialect\NorthStar\IR\NorthStarAttrs.cpp.inc中。

# 第四章.自定义Operation

​	先看一下5-define_operation\include\Dialect\NorthStar\IR\NorthStarTypes.td，这个是我们自定义类型的tablegen文件。新加了一个NorthStar_BufferType类型。

```tablegen
// 定义 BufferType，继承 NorthStar_Type，C++ 名称 "Buffer"，缩写 "buffer"。
def NorthStar_BufferType : NorthStar_Type<"Buffer","buffer",[]>{
  let summary = " the summary of north-star buffer type";	// 简要描述和详细描述。

  let description = "description of north-star buffer type";

  // 构造参数只有一个:ArrayRefParameter<"int64_t">:$devices：一个设备ID数组，表示 Buffer 绑定的多个设备。
  let parameters = (ins 
    ArrayRefParameter<"int64_t">:$devices
  );

  let genStorageClass = 1;	//生成 Storage 类，不自定义构造。
  let hasStorageCustomConstructor = 0;

  let assemblyFormat = "`<`$devices`>`";	//assembly 格式定义，打印和解析时格式是 <devices>，例如 <0,1,2> 表示设备0、1、2。

  let skipDefaultBuilders = 0;		// 不跳过默认构造器。

  let genVerifyDecl = 1;		// 生成 verify 声明。
}
```

​	然后是6-define_interface\include\Dialect\NorthStar\IR\NorthStarConstraints.td，这是对类型的约束。

```
// 定义了一个名为 AnyNSTensor 的类型约束（def是 TableGen 定义关键字）。继承于 Type<...>，表示这是一个类型约束的定义。
// And<[ ... ]>：表示多个条件的逻辑“与”，所有条件必须同时满足。CPred<"条件表达式">：用 C++ 表达式作为谓词断言，返回 true 或 false，$_self 是当前类型对象的占位符。
// ::mlir::isa<::mlir::north_star::NSTensorType>($_self)	这是调用 LLVM/MLIR 的 isa<T>(obj) 模板函数，判断 $_self 是否是 NSTensorType 类型或其子类。
// ::mlir::cast<::mlir::north_star::NSTensorType>($_self).getShape().size() > 0		cast首先将 $_self 强制转换（cast）为 NSTensorType。	然后调用 getShape() 方法，返回形状（shape）信息，一般是维度列表。	.size() > 0 表示该张量类型必须有非空的 shape，至少有一个维度。

def AnyNSTensor : Type<And<[CPred<"::mlir::isa<::mlir::north_star::NSTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::north_star::NSTensorType>($_self).getShape().size() > 0">]>>;


// 与上面注释类似
def AnyBuffer  : Type<And<[CPred<"::mlir::isa<::mlir::north_star::BufferType>($_self)">,
                            CPred<"::mlir::cast<::mlir::north_star::BufferType>($_self).getDevices().size() > 0">]>>;
```

​	自定义op的tablegen文件路径在5-define_operation\include\Dialect\NorthStar\IR\NorthStarOps.td

```tablegen
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef DIALECT_NORTH_STAR_OPS_TD
#define DIALECT_NORTH_STAR_OPS_TD

include "Dialect/NorthStar/IR/NorthStarAttrs.td"


// 自定义op基类，继承自 Op
// 该类定义了op的基本属性和行为
// mnemonic 是操作符的名字（如 "add"）
class NorthStar_Op<string mnemonic, list<Trait> traits = []>
    : Op<NorthStar_Dialect, mnemonic, traits> {
  // 拼接字符串用于文档注释，例如 "north_star::AddOp op"
  let summary = cppNamespace#opName#"op";

  let description = "$_name op";
}

// 单目op
// mnemonic,操作名称（如 "exp"）    OperandType：输入类型（如 AnyNSTensor）
// resultType：默认与输入类型相同   traits：支持附加特性（比如 NoSideEffect）   attributes：额外参数，例如 axis。
class NorthStar_UnaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_Op<mnemonic, traits#[]>{
    // 表示操作有一个输入 input，类型是 OperandType；同时拼接上额外的 attributes。
    let arguments = !con((ins
        OperandType:$input),
        attributes
        );

    // 表示操作有一个输出 result，类型是 resultType。
    let results = (outs
        resultType:$result);
}

//双目op
class NorthStar_BinaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_Op<mnemonic, traits#[]>{
    // 表示操作有两个输入 lhs(左操作数) 和 rhs(右操作数)，类型都是 OperandType；同时拼接上额外的 attributes。
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
        );

    let results = (outs
        resultType:$result);
}

// 常量操作     接收一个 ElementsAttr 类型的属性作为值；输出一个 AnyNSTensor 类型结果。
def NorthStar_ConstOp : NorthStar_Op<"const",[]>{
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnyNSTensor:$result);
}

def NorthStar_SoftmaxOp : NorthStar_UnaryOp<"softmax",AnyNSTensor, AnyNSTensor, [], (ins I64Attr:$axis)>{
    let hasVerifier = 1;    // cpp 文件中实现 verify() 方法。
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis) , 
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
            }]>

    ];
}

// 一元指数操作 接收一个 AnyNSTensor，返回同类型；
def NorthStar_ExpOp : NorthStar_UnaryOp<"exp",AnyNSTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input) , 
            [{
                $_state.addOperands(input);
                $_state.addTypes(input.getType());
            }]>
    ];
}

// 四则运算
def NorthStar_AddOp : NorthStar_BinaryOp<"add",AnyNSTensor>;
def NorthStar_SubOp : NorthStar_BinaryOp<"sub",AnyNSTensor>;
def NorthStar_MulOp : NorthStar_BinaryOp<"mul",AnyNSTensor>;
def NorthStar_DivOp : NorthStar_BinaryOp<"div",AnyNSTensor>;


def NorthStar_AllToAllOp : NorthStar_Op<"all_to_all",[]>{
    let arguments = (ins
        AnyBuffer:$input,
        AnyBuffer:$output
    );
}

// 接收若干个 Tensor，返回一个 Buffer
def NorthStar_BufferOp : NorthStar_Op<"buffer",[]>{
    let description = "将 多个device_id 的tensor 组合成 一个 buffer";
    let arguments = (ins
        Variadic<AnyNSTensor>:$tensors		//输入是多个AnyNSTensor类型的张量
    );
    let results = (outs
        AnyBuffer:$result);
    let hasVerifier = 1;
    
    // <(ins "::mlir::ValueRange":$tensors),[{ ... }]> 指明了构造函数的参数：输入参数是一个类型为 ::mlir::ValueRange 的变量，名字是 $tensors。       // ValueRange是MLIR 中表示一组 Value（操作的输入张量）的类型，可以看成张量的数组或序列。后面大括号里的 { ... } 是构造器的具体实现代码。
    // $_state.addOperands(tensors)；$_state 是一个内置变量，表示“正在构造的操作状态”（OperationState 对象）。addOperands(tensors) 是把输入的       // $tensors（多个张量）作为操作的操作数（Operands）加入当前操作。这一步把所有传入的张量“绑定”给操作，操作的输入就确定了。
    // ::llvm::SmallVector<int64_t> devices;定义了一个空的 SmallVector<int64_t> 容器，命名为 devices。SmallVector 是 LLVM 提供的类似动态数组的     // 容器。这个容器用来存储每个输入张量对应的设备编号（device_id）。
    // for (auto tensor : tensors) { ... }	遍历所有输入张量 $tensors。auto tensor 是循环变量，表示当前遍历的单个张量（类型是 mlir::Value）。
    // auto tensor_type = llvm::cast<::mlir::north_star::NSTensorType>(tensor.getType());
    // tensor.getType()：获得当前张量的类型，类型是 mlir::Type。
    // llvm::cast<...>(...)：安全地将类型转换为特定子类型 NSTensorType，这是 NorthStar 方言自定义的张量类型。
    // 这里断言输入的每个张量的类型都是 NSTensorType，否则会报错。
    // devices.push_back(tensor_type.getDeviceId());从 NSTensorType 中获取设备ID（getDeviceId() 返回一个 int64_t）。把这个设备ID加入 devices       // 容器，收集所有输入张量对应的设备ID。
    // $_state.addTypes(::mlir::north_star::BufferType::get($_state.getContext(), devices));
    // 构造一个 BufferType，它是 NorthStar 方言中表示设备 buffer 的类型。
    // 调用静态方法 BufferType::get：传入当前操作的上下文（$_state.getContext()）	传入刚刚收集的设备ID列表（devices）
    // 把生成的 BufferType 类型加入到当前操作的结果类型列表中（这个操作只有一个结果）。
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors) , 
            [{
                $_state.addOperands(tensors);
                ::llvm::SmallVector<int64_t> devices;
                for (auto tensor : tensors) {
                    auto tensor_type =
                        llvm::cast<::mlir::north_star::NSTensorType>(tensor.getType());
                    devices.push_back(tensor_type.getDeviceId());
                }
                $_state.addTypes(::mlir::north_star::BufferType::get($_state.getContext(), devices));
            }]>
    ];
}

def NorthStar_GetTensorOp: NorthStar_Op<"get_tensor",[]>{
    let description = "从buffer中取出指定device_id的tensor";
    let arguments = (ins
        AnyBuffer:$buffer,
        I64Attr:$device_id
    );
    let results = (outs
        AnyNSTensor:$result);
    let hasVerifier = 1;
}

// 通用打印操作；   输入可以是任何类型；
def NorthStar_PrintOp: NorthStar_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}


#endif // INCLUDE_LLCOMPILER_DIALECT_NorthStar_IR_LLHOPS_TD_

```

# 第五章.自定义Interface

## 5.1内建Interface	

​	在 **MLIR（Multi-Level Intermediate Representation）** 中，`Interface` 是一种 **可插拔的机制**，用于给 **类型（Type）**、**属性（Attribute）** 或 **操作（Operation）** 增加一组统一的功能或行为约定。它的核心作用是 **通过抽象定义一组方法，让多个类实现相同的接口，从而实现行为上的统一和灵活扩展**，这和 C++ 的虚函数或 Java 的接口类似，但在 MLIR 中更灵活、表意更强，并且支持 TableGen 自动生成代码。

​	MLIR中一些常用内建Interface如下。

​	Type Interfaces（类型接口）

| 接口名                       | 用途说明                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| `ShapedTypeInterface`        | 表示具有“形状”的类型（如 Tensor、MemRef），提供维度、元素类型、动态维度等接口 |
| `MemRefElementTypeInterface` | 表示可以作为 MemRef 元素的类型（例如整数、浮点类型等）       |
| `TypedTypeInterface`         | 表示带有类型的类型，用于支持 `getElementType()` 等操作       |
| `SubElementTypeInterface`    | 支持对子类型进行递归访问与替换，用于分析或转换嵌套类型结构   |

​	Attribute Interfaces（属性接口）

| 接口名                      | 用途说明                                                   |
| --------------------------- | ---------------------------------------------------------- |
| `TypedAttrInterface`        | 表示拥有类型的 Attribute（如 DenseElementsAttr）           |
| `ElementsAttrInterface`     | 表示元素型 Attribute，支持以多种方式遍历元素值             |
| `MemRefLayoutAttrInterface` | 表示可用作 MemRef layout 的属性（支持 stride/offset 表达） |

​	Operation Interfaces（操作接口）

| 接口名                    | 用途说明                                                     |
| ------------------------- | ------------------------------------------------------------ |
| `InferTypeOpInterface`    | 支持根据操作输入自动推导输出类型                             |
| `MemoryEffectOpInterface` | 表示操作对内存的读写影响                                     |
| `RegionBranchOpInterface` | 用于表示操作中包含的控制流区域（如 if、while）               |
| `LoopLikeOpInterface`     | 表示类似于 for/while 的循环操作，暴露 loop 结构              |
| `CallOpInterface`         | 用于表示调用函数/操作的接口（如 call）                       |
| `ViewLikeOpInterface`     | 表示行为类似 view 的操作（形状更改但不复制数据）             |
| `ControlFlowInterfaces`   | 包括 `BranchOpInterface`、`BranchOpInterface`, 提供控制流跳转相关接口 |

## 	5.2自定义Interface

​	首先看一下BuiltinTypeInteface.td文件，路径为third_party\llvm-project\mlir\include\mlir\IR\BuiltinTypeInterfaces.td

```tablegen
//===- BuiltinTypeInterfaces.td - Builtin type interfaces --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for type interfaces that closely interact with
// attributes, types, and operations in the builtin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPEINTERFACES_TD_
#define MLIR_IR_BUILTINTYPEINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MemRefElementTypeInterface
//===----------------------------------------------------------------------===//

def MemRefElementTypeInterface : TypeInterface<"MemRefElementTypeInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Indication that this type can be used as element in memref types.

    Implementing this interface establishes a contract between this type and the
    memref type indicating that this type can be used as element of ranked or
    unranked memrefs. The type is expected to:

      - model an entity stored in memory;
      - have non-zero size.

    For example, scalar values such as integers can implement this interface,
    but indicator types such as `void` or `unit` should not.

    The interface currently has no methods and is used by types to opt into
    being memref elements. This may change in the future, in particular to
    require types to provide their size or alignment given a data layout.
  }];
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

def ShapedTypeInterface : TypeInterface<"ShapedType"> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface provides a common API for interacting with multi-dimensional
    container types. These types contain a shape and an element type.

    A shape is a list of sizes corresponding to the dimensions of the container.
    If the number of dimensions in the shape is unknown, the shape is "unranked".
    If the number of dimensions is known, the shape "ranked". The sizes of the
    dimensions of the shape must be positive, or kDynamic (in which case the
    size of the dimension is dynamic, or not statically known).
  }];
  let methods = [
    InterfaceMethod<[{
      Returns a clone of this type with the given shape and element type.

      If no shape is provided, the shape of this type is used. In that case, if
      this type is unranked, so is the resulting type.

      If a shape is provided, the resulting type is always ranked, even if this
      type is unranked.
    }],
    "::mlir::ShapedType", "cloneWith", (ins
      "::std::optional<::llvm::ArrayRef<int64_t>>":$shape,
      "::mlir::Type":$elementType
    )>,

    InterfaceMethod<[{
      Returns the element type of this shaped type.
    }],
    "::mlir::Type", "getElementType">,

    InterfaceMethod<[{
      Returns if this type is ranked, i.e. it has a known number of dimensions.
    }],
    "bool", "hasRank">,

    InterfaceMethod<[{
      Returns the shape of this type if it is ranked, otherwise asserts.
    }],
    "::llvm::ArrayRef<int64_t>", "getShape">,
  ];
  let extraClassDeclaration = [{
    /// 定义一个常量 kDynamic，用来表示“动态维度”的标记值，使用 int64_t 最小值来标识。
    static constexpr int64_t kDynamic =
        std::numeric_limits<int64_t>::min();

    /// Whether the given dimension size indicates a dynamic dimension.
    /// 判断某个维度值 dValue 是否是动态维度（即是否等于 kDynamic）。
    static constexpr bool isDynamic(int64_t dValue) {
      return dValue == kDynamic;
    }

    /// Whether the given shape has any size that indicates a dynamic dimension.
    /// 判断一个形状（数组）中是否包含至少一个动态维度。
    static bool isDynamicShape(ArrayRef<int64_t> dSizes) {
      return any_of(dSizes, [](int64_t dSize) { return isDynamic(dSize); });
    }

    /// Return the number of elements present in the given shape.
    /// 计算一个静态形状（所有维度均为常量）中的总元素个数，函数声明未定义，定义在 C++ 中。
    static int64_t getNumElements(ArrayRef<int64_t> shape);

    /// Return a clone of this type with the given new shape and element type.
    /// The returned type is ranked, even if this type is unranked.
    /// 基于新的 shape 和 elementType 生成一个副本（克隆），即使当前类型是 unranked，生成的类型也会是 ranked。
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape. The returned type
    /// is ranked, even if this type is unranked.
    /// 基于新的 shape 克隆当前类型，elementType 保持不变。返回类型同样是 ranked。
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return cloneWith(shape, getElementType());
    }
  }];

  let extraSharedClassDeclaration = [{
    /// Return a clone of this type with the given new element type. The
    /// returned type is ranked if and only if this type is ranked. In that
    /// case, the returned type has the same shape as this type.
    /// 基于新的 elementType 克隆当前类型。如果当前是 ranked，则保留形状；否则就不指定形状。
    auto clone(::mlir::Type elementType) {
      return $_type.cloneWith(/*shape=*/std::nullopt, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    /// 获取当前类型中元素类型（整型或浮点型）的位宽（bit 数）。若不是整数或浮点类型会报错。
    unsigned getElementTypeBitWidth() const {
      return $_type.getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    /// 获取当前类型的秩（rank），即维度个数。若是 unranked 会触发断言。
    int64_t getRank() const {
      assert($_type.hasRank() && "cannot query rank of unranked shaped type");
      return $_type.getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    /// 返回总元素个数，但仅限 static shape。如果有动态维度则会断言失败。
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements($_type.getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    /// 判断指定维度是否为动态大小。要求当前是 ranked 类型。
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic($_type.getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    /// 判断是否为静态形状，即要求是 ranked 且所有维度都不是动态。
    bool hasStaticShape() const {
      return $_type.hasRank() &&
             !::mlir::ShapedType::isDynamicShape($_type.getShape());
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    /// 不仅判断是否是静态形状，还要和给定 shape 完全一致。
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && $_type.getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    /// 统计当前类型中有多少个维度是动态的。要求是 ranked 类型。
    int64_t getNumDynamicDims() const {
      return llvm::count_if($_type.getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    /// 获取第 idx 个维度的大小（不论是否是动态），需确保类型为 ranked 且索引合法。
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return $_type.getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    /// 给定某个动态维度的索引 index，计算它在所有动态维度中的相对编号。
    /// 比如 shape 为 [2, ?, 5, ?]，其中 index 为 3（对应第二个 ?），它是所有动态维度中的第 2 个（返回 1）。
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if($_type.getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  }];
}

#endif // MLIR_IR_BUILTINTYPEINTERFACES_TD_

```

​	然后是在NorthStarTypes.td(路径为6-define_interface\include\Dialect\NorthStar\IR\NorthStarTypes.td)中添加额外成员函数声明。

```tablegen
/// 这里挂接了ShapedTypeInterface，即[ShapedTypeInterface]
def NorthStar_TensorType : NorthStar_Type<"NSTensor","ns_tensor",[ShapedTypeInterface],"::mlir::TensorType">{
  // 概述
  let summary = " the summary of north-star tensor type";

  // 方言的详细描述
  let description = "description of north-star tensor type";

  // 参数
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$device_id
  );

  // 是否生成StorageClass, 无特殊情况，建议设为ture
  let genStorageClass = 1;
  
  // 不建议改动
  let hasStorageCustomConstructor = 0;

  // 额外的builder 声明
  let builders = [
    TypeBuilder<(ins 
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType),[{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;
  
  /// 挂接了 ShapedType::Trait<T> 中的多个常用函数，这允许在 NSTensorType 实例上直接调用
  /// 例如NSTensorType ty = ...;
  /// ty.getRank();               // from ShapedType
  /// ty.getElementType();        // 来自接口或继承
  /// ty.getNumElements();
  let extraClassDeclaration = [{
  	/// 把 ShapedType::Trait<NSTensorType> 中的 getElementTypeBitWidth() 函数“引入”为 NSTensorType 的成员函数。获取元素类型的位宽
    using ::mlir::ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
    /// 返回张量的秩（rank），也就是 shape.size() 的值。
    using ::mlir::ShapedType::Trait<NSTensorType>::getRank;
    /// 返回元素个数。只对所有维度静态的 tensor 有意义（否则返回 std::nullopt）
    using ::mlir::ShapedType::Trait<NSTensorType>::getNumElements;
    /// 判断某个维度是不是动态维度（即该维度是 -1）
    using ::mlir::ShapedType::Trait<NSTensorType>::isDynamicDim;
    /// 判断所有维度是否都是静态的（即没有 -1）
    using ::mlir::ShapedType::Trait<NSTensorType>::hasStaticShape;
    /// 返回多少个维度是动态的（也就是 -1 的个数）
    using ::mlir::ShapedType::Trait<NSTensorType>::getNumDynamicDims;
    /// 获取指定维度的大小（可能是正整数或 -1）
    using ::mlir::ShapedType::Trait<NSTensorType>::getDimSize;
    /// 获取在某个维度是第几个“动态维度”。比如第 2 维是 -1，它可能是动态维度的第一个。
    using ::mlir::ShapedType::Trait<NSTensorType>::getDynamicDimIndex;

    /// 自定义的 clone() 方法族
    /// 克隆一个新的类型，可以选择替换 shape 和 elementType。如果 shape 是空的，就保留当前 shape。
    ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type type){
        if(shape)
          return mlir::north_star::NSTensorType::get(getContext(), *shape, type);
        return mlir::north_star::NSTensorType::get(getContext(), getShape(), type);
    }
    /// 克隆当前类型，但替换 shape 和 device_id，elementType 保持不变。
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape,int64_t device_id) const {
      return NSTensorType::get(getContext(), shape, getElementType(),device_id);
    }
    /// 克隆当前类型，替换 shape 和 elementType，device_id 保持不变
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) const {
      return NSTensorType::get(getContext(), shape, elementType,getDeviceId());
    }
    /// 只替换 shape，其他字段保持不变。
    NSTensorType clone(::mlir::ArrayRef<int64_t> shape) const {
      return NSTensorType::get(getContext(), shape, getElementType(),getDeviceId());
    }
    /// 只替换 elementType。
    NSTensorType clone(::mlir::Type elementType) const {
      return NSTensorType::get(getContext(), getShape(), elementType,getDeviceId());
    }
    /// 完全克隆自己（拷贝构造）。
    NSTensorType clone() const {
      return NSTensorType::get(getContext(), getShape(), getElementType(),getDeviceId());
    }
    }];
}
```

​		

​	下面是自定义Interface的tablegen文件，路径为6-define_interface\include\Interfaces\DistributeParallelismInterfaces.td

```
//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distribute under the License is distribute on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD
#define INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD
include "mlir/IR/Interfaces.td"

// 定义一个属性接口，名为DistributeParallelAttr
def DistributeParallelAttr: AttrInterface<"DistributeParallelAttr">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  let methods = [];		// 空数组，表示此接口目前没有任何定义的方法，仅作为一个基础标识符接口使用。
  let extraClassDeclaration = "";		// 空数组，表示此接口目前没有任何定义的方法，仅作为一个基础标识符接口使用。
  let extraSharedClassDeclaration = "";	// 空数组，表示此接口目前没有任何定义的方法，仅作为一个基础标识符接口使用。
}

// 定义继承自 DistributeParallelAttr 的属性接口 DataParallelAttr,通过[DistributeParallelAttr]指定
def DataParallelAttr: AttrInterface<"DataParallelAttr",[DistributeParallelAttr]>{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  
  // 定义了两个接口方法:
  // 1.getDPNums()：返回类型是 int64_t。注释说明是“DP 数量”，即数据并行的数量。
  // 2.getDevices():返回类型是 ::llvm::ArrayRef<int64_t>，即一组整数设备编号。注释说明是“设备编号”，表示参与并行计算的设备列表。
  // 这两个方法要求实现这个接口的 Attribute 类必须提供这两个成员函数。
  let methods = [
    InterfaceMethod<[{
      DP 数量.
    }],
    "int64_t", "getDPNums">,

    InterfaceMethod<[{
      设备编号.
    }],
    "::llvm::ArrayRef<int64_t>", "getDevices">,
  ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

// 定义支持数据并行的操作接口
def SupportedDataParallelismOp: OpInterface<"SupportedDataParallelismOp">{
  let description = "Properties related to data parallelism";
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "applyDataParallelism",
      /*args=*/        (ins "::mlir::DistributeParallelAttr":$attr),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return llvm::failure();
      }]
      >,
      InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "bool",
      /*methodName=*/  "supportedDataParallelism",
      /*args=*/        (ins),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      Operation* op = $_op.getOperation();
      if (op->getNumOperands() == 0) return true;
      auto base_type = op->getOperand(0).getType();
      if (!isa<mlir::ShapedType>(base_type)) return false;
      for (auto operand : op->getOperands()) {
        if (operand.getType() != base_type) return false;
      }
      return true;
      }]
      >
      ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

// 定义通用的分布式并行操作接口 DistributeParallelOp
def DistributeParallelOp:OpInterface<"DistributeParallelOp">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  let methods = [];
  let extraClassDeclaration = "";
  
  // 在 extraSharedClassDeclaration 中定义了两个 C++ 成员函数：
  // 1.	applyDistributeParallelism(const ::mlir::DistributeParallelAttr attr)：
  // 	该方法判断传入的 attr 是不是 DataParallelAttr。如果是，则判断该操作是否支持 SupportedDataParallelismOp 接口。
  // 	若支持，则调用 applyDataParallelism 完成转换。如果不是 DataParallelAttr 类型，触发错误 llvm_unreachable。最后返回失败。
  // 2.	supportedDistributeParallelism()：
  //	判断当前操作是否支持数据并行。通过尝试转换成 SupportedDataParallelismOp，调用它的接口方法。不支持则触发错误 llvm_unreachable。返回 false。
  let extraSharedClassDeclaration = [{
    // 实现并行变换
    ::mlir::LogicalResult applyDistributeParallelism(const ::mlir::DistributeParallelAttr attr){
      if (isa<mlir::DataParallelAttr>(attr)) {
        if (!isa<mlir::SupportedDataParallelismOp>($_op.getOperation())) return ::llvm::failure();
        return dyn_cast<mlir::SupportedDataParallelismOp>($_op.getOperation()).applyDataParallelism(attr);
      } else {
        llvm_unreachable("unsupported parallel type!");
      }
      return ::llvm::failure();
    };

    bool supportedDistributeParallelism(){
      if (isa<mlir::SupportedDataParallelismOp>($_op.getOperation())){
        return dyn_cast<mlir::SupportedDataParallelismOp>($_op.getOperation()).supportedDataParallelism();
      }else{
        llvm_unreachable("unsupported parallel type!");
      }
      return false;
    }
  }];
}

#endif // INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD
```

​	使用mlir_tablegen工具生成c++文件

```cmake
set(LLVM_TARGET_DEFINITIONS DistributeParallelismInterfaces.td)
mlir_tablegen(DistributeParallelismOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DistributeParallelismOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(DistributeParallelismAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(DistributeParallelismAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRDistributeParallelismInterfacesIncGen${ch_num})


```

​	通过main函数验证 `NSTensorType` 类型支持 `ShapedTypeInterface`，验证 `DataParallelismAttr` 属性支持接口，最后构建完整模块并执行遍历应用接口方法 `applyDistributeParallelism()`。下面代码为main文件中的CH6部分。

```cpp
void CH6() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  // shaped type interface
  auto f32 = mlir::Float32Type::get(&context);		//创建一个 f32 类型（浮点数类型）。
  auto dim = mlir::ShapedType::kDynamic;
  auto shape = mlir::SmallVector<int64_t>({dim, dim, 24});		//使用 kDynamic 表示前两个维度为动态维度（-1）。第三个维度是静态的 24。
  auto tensor_type =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 0);		//创建tensor_type，传入shape，elementtype f32,device_id 0三个参数
  
    //将tensor_type转为shaped_type类型。用于调用 .clone()、.getNumElements()、.getShape() 等通用接口方法。
  auto shaped_type = mlir::cast<mlir::ShapedType>(tensor_type);
  
  //dump() 是调试用方法，会把类型信息打印到 llvm::outs()。这里分别打印 NSTensorType 自身和转换为 ShapedType 后的信息。
  llvm::outs() << "NSTensorType: \t";
  tensor_type.dump();
  llvm::outs() << "Shaped Type Interface:\t";
  shaped_type.dump();
 
  //测试 ShapedTypeInterface 中的 clone(ElementType) 方法。会保持 shape/device 不变，仅替换元素类型为f32。
  auto cloned_type = shaped_type.clone(f32);
  llvm::outs() << "Cloned Shaped Type Interface:\t";
  cloned_type.dump();
  
  // Attr interface
  //创建一个自定义的 DataParallelismAttr，表示并行度为 2。
  auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
  
  // getAbstractAttribute().getName()：属性注册时的名字。hasInterface(...)：检查这个属性是否注册了某个接口，这里检查的是DistributeParallelAttr
  // DistributeParallelAttr::getInterfaceID()：接口 ID，全局唯一的标识符。这段语句输出为north_star.DP has mlir::DataParallelAttr interface: 1
  llvm::outs() << dp_attr.getAbstractAttribute().getName()
               << " has mlir::DataParallelAttr interface: "
               << dp_attr.getAbstractAttribute().hasInterface(
                      mlir::DistributeParallelAttr::getInterfaceID())
               << "\n";
  // 这为上面检测接口方法的另一种写法。
  llvm::outs()
      << dp_attr.getAbstractAttribute().getName()
      << " has mlir::DataParallelAttr interface: "
      << dp_attr.hasPromiseOrImplementsInterface<mlir::DataParallelAttr>()
      << "\n";

  // 调用自定义的函数 getModule() 构造一个模块，包含：
  // 一个名为 "main" 的函数（func.func)
  // 函数参数/返回值是 NSTensorType<?x?x24xf32, 0>
  // 函数上挂了 dp_attr 属性
  // 函数体中有两个 SoftmaxOp 和一个 func.return
  // 输出结果为
  // "builtin.module"() <{sym_name = "NorthStar"}> ({
  // "func.func"() <{function_type = (!north_star.ns_tensor<128x128x24xf32,0>) -> !north_star.ns_tensor<128x128x24xf32,0>, sym_name =    /////"main"}> ({
  // ^bb0(%arg0: !north_star.ns_tensor<128x128x24xf32,0>):
  // %0 = "north_star.softmax"(%arg0) <{axis = 1 : si64}> : (!north_star.ns_tensor<128x128x24xf32,0>) -> /////!north_star.ns_tensor<128x128x24xf32,0>
  // %1 = "north_star.softmax"(%0) <{axis = 1 : si64}> : (!north_star.ns_tensor<128x128x24xf32,0>) -> /////!north_star.ns_tensor<128x128x24xf32,0>
  // "func.return"(%1) : (!north_star.ns_tensor<128x128x24xf32,0>) -> ()
  // }) {dp_attr = #north_star.DP<DP = 2 : 0, 1>} : () -> ()
  // }) : () -> ()
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  module->dump();
  //遍历整个 module，找到所有 func::FuncOp
  module->walk([](mlir::func::FuncOp func) {
    // 如果这个函数拥有名为 "dp_attr" 的 Attribute，并且它实现了接口 DistributeParallelAttr，则继续
    if (auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
            func->getAttr(KDPAttrName))) {
      // 遍历这个函数体内的操作，找出实现了接口 DistributeParallelOp 的 Op
      func->walk([&](mlir::Operation* op) {
        if (auto dis_op =
                llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
          // 调用接口方法 applyDistributeParallelism：这是在 InterfaceMethod 中定义的方法。如果返回 success()，说明成功“变换”。然后删除这个 op
          if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
            llvm::outs() << "Apply DataParallelism to " << op->getName()
                         << "\n";
            op->erase();
          };
        }
      });
    }
  });
  module->dump();
}
int main() { CH6(); }
```

# 第六章.IR结构和基本数据结构

​	MLIR 中的 IR（Intermediate Representation，中间表示）结构是整个编译器基础的核心。

​	MLIR 的 IR 是一个树状结构（多层嵌套），如下图所示：

```
ModuleOp
 └── Region(s)
      └── Block(s)
           └── Operation(s)
                └── Region(s) (可嵌套)

```

​	1.ModuleOp：顶层容器，相当于程序的“根”，它是一个特殊的 Operation，代表一个程序模块，可以包含多个 Region。

​	2.Region：是一个控制流区域，包含一个或多个 Block，Operation 可以携带 Region，用来表示作用域、函数体、循环体等。

​	3.Block：是一组顺序执行的操作集合（Operation），拥有自己的参数（Block Argument）和 terminator（控制流终结指令，如 return）。

​	4.Operation：最基本的执行单位，可以是计算、调用、内存操作等。

​	例如

```
module {		//ModuleOp，是 MLIR IR 的顶层操作。		{ : 表示 module 包含一个 Region。这个 Region 会包含 Block，Block 中是顺序执行的操作。
  func @main() -> i32 {		//func : 声明一个函数。MLIR 标准方言中的 FuncOp。		@main : 函数的符号名称，等价于 llvm::Function::Name。									() : 函数参数为空（没有输入参数）。	-> i32 : 函数返回类型是一个 i32。		{: 函数体，是 FuncOp 的 Region。
    %0 = constant 42 : i32		//%0: 这个是一个 SSA 值的名称，表示这个操作的结果被命名为 %0。	constant: 是一个操作,表示创建一个常量值。
    return %0 : i32		// 返回%0，返回类型为i32
  }		// 函数体结束
}		// 模块结束
```

​	MLIR 的 IR 实际上由以下六种基本元素组成：

| 元素         | 说明                       |
| ------------ | -------------------------- |
| 1. Operation | 操作（基本执行单元）       |
| 2. Region    | 区域（控制流域）           |
| 3. Block     | 块（指令集合）             |
| 4. Value     | SSA 值（操作的输入和输出） |
| 5. Type      | 类型（Value 的类型）       |
| 6. Attribute | 属性（编译时常量元数据）   |

​	1.Operation : MLIR 的所有执行语义都通过 `Operation` 表达。比如操作名：如 `std.addf`, `func.return`, `mydialect.softmax`	操作数：输入值（Value）			      结果：输出值（Value）	类型：操作数和结果都有类型（Type）	属性（Attributes）：附加信息，如 axis, name, shape 等	匿名/命			      名 Region	（Location）源代码映射信息

​	2.Region ： Region 是一个控制流作用域,包含一个或多个 Block,一个 Operation 可以有多个 Region（如 `func`, `if`, `while`）。

​	3.Block ：Block 是一个 顺序执行的指令列表，并带有 SSA 参数和一个 terminator。Block 内容有Block 参数（参数是 Value）、Operation 序列（中间指令）、			Terminator Operation（最后一条必须是终止指令）

​	4.Value（SSA 值）：所有操作的输入和输出都是 `Value`。Value 是一个 **SSA 值**（Single Static Assignment），只定义一次，可以使用多次。
