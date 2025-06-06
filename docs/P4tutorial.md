# p4 Tutorial

相关教程：

- https://github.com/nsg-ethz/p4-learning （主要）
- https://github.com/p4lang （文档等）
- https://google.com （中文社区质量较低，一般不容易搜到解决方案）

网络技术相关的资源：

- https://feisky.gitbooks.io/sdn/content/

---



## 1. p4 环境安装 

**OS**：ubuntu 20.04.4

**depends：**

- **PI** ：p4runtime api，网络拓扑中有p4交换机时，必须安装该模块；
- **BMv2**： p4交换机虚拟机 ；
- **P4C**：p4程序编译器，支持p4_14 & p4_16；
- **Mininet**：基于namespace的linux网络仿真软件，对外提供python api；
- **FRRouting** ：网络协议栈仿真软件；
- **P4-Utils**：

**可优先使用安装脚本安装**：`https://github.com/nsg-ethz/p4-utils/blob/master/install-tools/install-p4-dev.sh`

PI、BMv2需要编译安装，有一定的安装难度。

P4C、Mininet、FRRouting支持用包管理器安装，非常容易操作。



### 1.1 PI 安装

> 建议每一个模块都在home路径下新建一个文件夹存放文件。
>
> PI部分安装较为繁琐，如有特殊报错需根据本地具体环境进行查阅

#### 1.1.1 子模块安装

##### 1.1.1.1 无需编译模块

```bash
apt install libreadline-dev valgrind libtool-bin libboost-dev libboost-system-dev libboost-thread-dev
```



##### 1.1.1.2 prtobuf v3.18.1

> https://github.com/p4lang/PI 	*protbuf部分*

- 安装步骤：

```bash
cd #回到home
git clone --depth=1 -b v3.18.1 https://github.com/google/protobuf.git
cd protobuf/
./autogen.sh
./configure
make
[sudo] make install
[sudo] ldconfig
```

- 可能出现的问题：
  显示未安装googletest, 或是警告No configuration information is in third_party/googletest in version1.10
  解决方法：https://github.com/google/protobuf.git 从这个上面找到thrid_party这个文件夹，然后找到里面的googletest文件夹并将其中的文件下载下来，放入/protobuf/thrid_party/googletest文件夹下，便可解决问题


##### 1.1.1.3 gRPC v1.43.2

> https://github.com/p4lang/PI 	*grpc部分*

- 安装步骤：

```shell
apt-get install build-essential autoconf libtool pkg-config
apt-get install cmake
apt-get install clang libc++-dev
apt-get install zlib1g-dev

cd #回到home
git clone --depth=1 -b v1.43.2 https://github.com/google/grpc.git 
cd grpc
git submodule update --init --recursive 
mkdir -p cmake/build
cd cmake/build
cmake ../..
make
make install
ldconfig
```

- 可能出现的问题：

 1. `git submodule update --init --recursive` 失败

    网络原因，一直重复直到全部成功（比较费事间，建议同步执行 1.1.1.4 bmv2及其依赖），也可以将github中的grpc库clone到gitee，再从gitee clone；

 1. 按照https://github.com/p4lang/PI grpc部分安装会提示不支持make编译，建议用cmake

    参考上述安装步骤即可；



##### 1.1.1.4 bmv2依赖

> https://github.com/p4lang/behavioral-model/blob/main/README.md

- 安装步骤：

```bash
cd #回到home
git clone https://github.com/p4lang/behavioral-model.git

sudo apt-get install -y automake cmake libgmp-dev \
    libpcap-dev libboost-dev libboost-test-dev libboost-program-options-dev \
    libboost-system-dev libboost-filesystem-dev libboost-thread-dev \
    libevent-dev libtool flex bison pkg-config g++ libssl-dev
    
cd ci
[sudo] chmod +x install-*
[sudo]./install-nanomsg.sh
[sudo]./install-thrift.sh

./autogen.sh
./configure
make
[sudo] make install   # if you need to install bmv2
```

- 可能出现的问题：

 1. `git clone https://github.com/p4lang/behavioral-model.git`失败

    网络问题，重复执行git clone直到成功。



##### 1.1.1.4 sysrepo
###### 1.1.1.4.1 子模块 libyang 编译安装

> https://github.com/CESNET/libyang

- 步骤

```bash
 cd #回到home
 git clone --depth=1 -b v0.16-r1 https://github.com/CESNET/libyang.git
  cd libyang
  mkdir build
  cd build
  cmake ..
  make
  make install
```
- 可能出现的问题：
 1. 缺少 pcre：

  ```bash
  sudo apt-get update
  sudo apt-get install libpcre3 libpcre3-dev
  # or
  sudo apt-get install openssl libssl-dev
  ```


###### 1.1.1.4.2 本体编译安装
> https://github.com/p4lang/PI/blob/main/proto/README.md

- 安装步骤：

```bash
cd #回到home
git clone --depth=1 -b v0.7.5 https://github.com/sysrepo/sysrepo.git
cd sysrepo
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=Off -DCALL_TARGET_BINS_DIRECTLY=Off ..
make
[sudo] make install
```

-----------------------#目前仍不确定是否安装完成

- 可能会出现的问题：

 1. 该部分可能出现的问题很多，主要集中在执行`cmake -DCMAKE...`阶段，会出现缺少库的问题

    解决方法，如报错缺少xxx，执行`apt install xxx`；

    如果报错提示无法定位到xxx库，执行`apt install libxxx-dev；`

    如果仍然找不到该库，百度&google搜ubuntu安装xxx；

-  可能需要执行如下命令：
   sudo apt install libpython2-dev
   sudo apt install liblua5.1-0
   sudo apt-get install lua5.1-0-dev
   sudo apt install swig
   sudo apt install libavl-dev  #这句不太确定是否正确
   sudo apt-get install libev-dev
   sudo apt install python3-virtualenv

   redblack的安装（下载地址：https://sourceforge.net/projects/libredblack/files/）
   下载完成后直接对tar.gz文件进行解压，再进行安装即可，需要可以检查更新

   Cmocka的安装（下载地址：https://cmocka.org/files/1.1/）
   下载完成后解压，然后创建build文件夹，再./configure，之后按照里面INSTALL文件的说明进行安装


     直到cmake成功



#### 1.1.2 pi 编译安装

> https://github.com/p4lang/PI

- 安装步骤

```bash
cd #回到home

git clone https://github.com/p4lang/PI.git
cd PI
git submodule update --init --recursive
./autogen.sh
./configure --with-proto --with-bmv2 --with-cli
make
make check
[sudo] make install
```

- 可能出现的问题

 1. 编译时报错缺少xxx头文件

    解决方法同1.1.1.4.2 sysrepo部分；

 2. 执行`git submodule update --init --recursive` 比较费时间，建议同步安装p4c或者mininet



### 1.2.1 bmv2 安装

如在1.1.1.4 bmv2依赖 部分执行了`[sudo] make install` 那么该部分可以跳过，否则返回 1.1.1.4 bmv2依赖 部分执行相关操作。



### 1.2.2 P4C 安装

> https://github.com/p4lang/p4c

- 安装步骤

```bash
sudo apt-get install cmake g++ git automake libtool libgc-dev bison flex \
libfl-dev libgmp-dev libboost-dev libboost-iostreams-dev \
libboost-graph-dev llvm pkg-config python3 python3-pip \
tcpdump

pip3 install ipaddr scapy ply
sudo apt-get install -y doxygen graphviz texlive-full

安装方法1：
. /etc/os-release
echo "deb http://download.opensuse.org/repositories/home:/p4lang/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/home:p4lang.list
curl -L "http://download.opensuse.org/repositories/home:/p4lang/xUbuntu_${VERSION_ID}/Release.key" | sudo apt-key add -
sudo apt-get update
sudo apt install p4lang-p4c
```

```
安装方法2：
git clone --recursive https://github.com/p4lang/p4c.git
mkdir build
cd build
cmake .. <optional arguments>
make -j4
make -j4 check  
（check需要100成功，如果出现问题参考/home/wly/p4c/build/Testing/Temporary/LastTest.log的这个文件）
```

- 可能需要额外执行的命令：
  sudo apt-get install scapy
  pip install thrift
  1.若在LastTest.Log中发现   ImportError: cannot import name 'Thrift' from 'thrift' (unknown location) ubuntu  这个错误
  则卸载thrift（pip uninstall thrift）, 然后重新安装

  2.若在LastTest.Log中发现  /usr/bin/ld: cannot find /home/wly/p4c/backends/ebpf/runtime/usr/lib64/libbpf.a: No such file or directory  这个错误
  则表明需要安装 libbpf，在 p4c 文件夹下运行 python3 backends/ebpf/build_libbpf

- 可能出现的问题
 1. `sudo apt-get install -y doxygen graphviz texlive-full` 非常费时间，建议与编译gRPC或者编译prtobuf同时进行



### 1.2.3 mininet 安装

```bash
sudo apt install mininet
```



### 1.2.4 FRRouting 安装

> https://deb.frrouting.org

- 安装步骤：

```bash
# add GPG key
curl -s https://deb.frrouting.org/frr/keys.asc | sudo apt-key add -

# possible values for FRRVER: frr-6 frr-7 frr-8 frr-stable
# frr-stable will be the latest official stable release
FRRVER="frr-stable"
echo deb https://deb.frrouting.org/frr $(lsb_release -s -c) $FRRVER | sudo tee -a /etc/apt/sources.list.d/frr.list

# update and install FRR
sudo apt update && sudo apt install frr frr-pythontools
```

- 可能出现的问题

 1. apt update报错

    删除`/etc/apt/sources.list.d/frr.list`，执行`sudo apt update && sudo apt install frr frr-pythontools`



### 1.2.5 p4-utils

> https://github.com/nsg-ethz/p4-utils

- 安装步骤：

```bash
cd #回到home
git clone https://github.com/nsg-ethz/p4-utils.git
cd p4-utils
sudo ./install.sh

cd
git clone https://github.com/mininet/mininet mininet
cd mininet
# Build mininet
sudo PYTHON=python3 ./util/install.sh -nwv

apt-get install bridge-utils
```

- 可能遇到的问题：
 1. `./install.sh` 部分报错缺少xxx库，参考1.1.1.4.2 sysrepo部分；



### 1.3 运行时出现的BUG

> Under maintenance……



#### 1.3.1 无法调用xterm

- 报错：

> xterm: Xt error: Can't open display: %s
> xterm: DISPLAY is not set



- **原因**：https://github.com/mininet/mininet/wiki/FAQ#x11-forwarding

> 没有正确开启**X11 forwarding**



- **MAC OS X 解决方法**：https://zhuanlan.zhihu.com/p/265207166（下载XQuartz）

```zsh
$ brew install XQuartz
$ XQuartz
$ export DISPLAY=:0

$ ssh -Y root@xxx.xxx.xxx.xxx

#连接linux服务器端
$ xterm
```
	1. `export DISPLAY=:0` 仅对当前shell起作用.
	1. 最简单的办法：不安装X11，直接再多开一个shell




---
## 2. P4基础知识



### 2.1 p4程序基本结构（组件）介绍

> 参考资料：
>
> - 官方文档：
    >   - https://p4.org/specs/
> - 网络&博客文档：
    >   - https://www.sdnlab.com/17882.html
>   - https://www.zhihu.com/column/c_1336207793033015296
>   - https://bbs.huaweicloud.com/blogs/288890
>   - http://www.nfvschool.cn
> - 一些重要的文档：
    >   - BMv2中一些参数定义的介绍：https://github.com/nsg-ethz/p4-learning/wiki/BMv2-Simple-Switch#creating-multicast-groups

- 首部（Headers）
- 解析器（parsers）
- 表（tables）
- 动作（actions）
- 控制器（control）

> 该部分内容请详细阅读参考资料，参考资料中**sdnlab的文章**中详细介绍了p4基本语法、p4各个组件的功能，建议与参考资料中**知乎的文章**一起阅读，互相借鉴理解。nfvschool的p4文章对各个模块的总结也很到位，非常具有参考价值。

---



### 2.2 P4程序解读

> 该部分主要参考苏黎世联邦理工学院的*Advanced Topics in Communication Networks* lecture：https://github.com/nsg-ethz/p4-learning/tree/master/examples
> 里面的例子都非常有代表性，建议仔细理解，本节仅分析综合度相对高的几个实例。
>
> 该部分每小节开头都会贴出对应内容的完整源码，之后会对源码逐段分析（或对重点函数进行分析）。阅读时可以先大致浏览一遍源码，了解其大致结构及逻辑，然后再对照后续的讲解进行理解。由于我们的水平有限，程序解读可能会存在一定的差错，如存在任何歧义，请以官方文档中的定义及描述为主。

---



#### 以Source Routing为例

> https://github.com/nsg-ethz/p4-learning/tree/master/examples/source_routing
>
> 源路由实例是通过在数据包包头添加源路由字段（用于指定数据包需要经过的交换机节点），p4交换机解析源路由字段并判断如何转发数据包，从而实现数据包指定路径转发。

##### (1) include/header.p4

```c
#define MAX_HOPS 127

const bit<16> TYPE_IPV4 = 0x800;
const bit<16> TYPE_SOURCE_ROUTING = 0x1111;
...

header ethernet_t {
	...
}

header source_routing_t {
    bit<1> last_header;
    bit<7> switch_id;
}

header ipv4_t {
	...
  ...
}

struct metadata {
    /* empty */
}

struct headers {
    ethernet_t   ethernet;
    source_routing_t[MAX_HOPS] source_routes;
    ipv4_t       ipv4;
} 
```

p4程序定义数据包头部是用header分别定义各个字段的头部模版，最后再在结构体headers中使用报文头部模版实例化各个报文头部，这里的header可以理解为c语言中的struct。

ethernt头部和ipv4头部定义比较基础，这里不再过多赘述，我们将重点分析source_routing_t和headers的定义。

- **source_routing_t**

```c
header source_routing_t {
    bit<1> last_header;
    bit<7> switch_id;
}
```

首先分析 `header source_routing_t` ，里面包括两个变量：last_header：用于判断当前报文头部是否是最后一个头部；switch_id：用于判断当前交换机是否是我们制定路径中的指定交换机。

- **headers**

```c
#define MAX_HOPS 127
...
...
struct headers {
    ethernet_t   ethernet;
    source_routing_t[MAX_HOPS] source_routes;
    ipv4_t       ipv4;
} 
```

然后还需要注意的是`struct headers` 中的 `source_routing_t[MAX_HOPS] source_routes;` 在程序的开头定义了宏`#define MAX_HOPS 127` ，所以整个数据包包头我们可以理解为：

`ethernet + source_routers[0] + source_routers[1] + ... + source_routers[n] + ipv4`

这里的n由我们设定的路径决定，最多能经过128个交换机。使用 https://github.com/nsg-ethz/p4-learning/tree/master/examples/source_routing 的send.py程序可以创建一个包含源路由头部的数据包，具体使用方法参考链接。假设在执行send.py后输入 2 3 2 2 1，那么我们会得到一个这样的包头：

```c
ETH	| 0 2 | 0 3 | 0 2 | 0 2 | 1 1 | IPV4 |
  	   ↓     ↓     ↓     ↓     ↓
  	 SR[0] SR[1] SR[2] SR[3] SR[4]
```

可以看到SR[0]到SR[3]的last_header都是0（两个数字中的前一个），SR[4]的last_header是1。

在parser阶段，每一次执行extract，指针就指向下一报文头部，在执行一定次数extract后，指针最后指向ipv4头部对ipv4进行解析，最终accept该数据包。

---



##### (2) include/parser.p4

```c
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {

        transition parse_ethernet;

    }

    state parse_ethernet {

        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType){
            TYPE_IPV4: parse_ipv4;
            TYPE_SOURCE_ROUTING: parse_source_routing;
            default: accept;
        }
    }

    state parse_source_routing {
        packet.extract(hdr.source_routes.next);
        transition select(hdr.source_routes.last.last_header){
            1: parse_ipv4;
            default: parse_source_routing;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {

        //parsed headers have to be added again into the packet.
        packet.emit(hdr.ethernet);
        packet.emit(hdr.source_routes);
        packet.emit(hdr.ipv4);

    }
}
```

parser部分可以理解为实现了把输入数据包的头部剥离出来的功能，parser本质上是一个由多种状态组成的状态机。所有数据包的状态都从start状态出发，根据当前报头的TYPE字段不同从而转移到不同的状态下进行解析；而deparser部分（需要注意deparser本身属于conrtol，不是paser）会将剥离的报头重新添加到数据包中。

我们先看**Myparser的函数定义**，理解函数定义可以便于我们理解p4程序的处理逻辑。

```c
// Myparser的函数定义
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
  ...
  ...
}
```

- **packet_in**：packet_in是定义在core.p4中的extern模版，packet_in实例化的对象packet中包含了当前到达交换机的数据包中的信息，需要注意的是由packert_in实例化的对象packet不需要使用in、out或者inout等关键字修饰，它本身属于in类型的对象。而后面三个形参都在定义时声明了out、inout的属性。在p4中，被in修饰的参数可以理解为仅可读，被out修饰的对象可以理解为仅可写，inout为可读可写。

- **headers**：headers是在header.p4中定义的数据包头部结构体。`out headers hdr` 中声明的变量hdr是一个out类型、且数据结构为header类型的变量，hdr用来存放在parser过程中解析packet得到的数据。

- **metadata**：metadata是在header.p4中定义的用户自定义数据结构（通常用来存放寄存器、计数器与及Meter的数据）。后续的几个例子会介绍该字段的详细用法。

- **standard_metadata_t**：standard_metadata_t是在交换机runtime定义的数据结构，用来存储数据包转发时的ingress_port、packet_length、egress_spec、egress_port等信息。我们这里是用的p4 runtime是BMv2，standard_metadata_t的详细定义可以参考：https://github.com/p4lang/behavioral-model/blob/main/docs/simple_switch.md ，不同厂商的p4设备standard_metadata_t定义可能不同，因此需要根据具体的runtime定义设计p4程序。

接下来分析**MYparser中的各个状态**。状态转移过程决定了数据包包头的处理顺序，我们需要仔细分析各个状态下的parser分别做了什么处理，才能理解最后得到了一个什么样的数据包。

- **start**

```c
// start状态
state start {
  transition parse_ethernet;
}
```

start状态不区分数据包类型，直接将所有数据包状态转移到parse_ethernet状态。通常每个p4程序的第一个parser都是start状态，整个parser过程由start状态开始，以accept或reject结束。

- **parse_ethernet**

```c
// parse_ethernet状态
state parse_ethernet {
  packet.extract(hdr.ethernet);
  transition select(hdr.ethernet.etherType){
    TYPE_IPV4: parse_ipv4;
    TYPE_SOURCE_ROUTING: parse_source_routing;
    default: accept;
  }
}
```

parse_ethernet：parse_ethernet状态下，程序首先调用packet的extract函数，将packet中index指针指向的数据块提取出来（首先计算需要提取的比特数目n，然后将packet当前index位置后n个bit提取出来存入hdr的index指向的位置，此处需要注意hdr为out类型，仅可以写入），然后packet的index指针后移到下一个报文头部的首字节，并同时操作hdr.ethernet的index指针后移一位，整个过程如下所示（官方文档中的extract函数定义），所以extract函数可以理解为 *即改变了header的index，又改变了hdr的index*。

```c
// extract函数定义（伪代码）
void packet_in.extract<T>(out T headerLValue) { 
  bitsToExtract = sizeofInBits(headerLValue);
  lastBitNeeded = this.nextBitIndex + bitsToExtract; 
  ParserModel.verify(this.lengthInBits >= lastBitNeeded, error.PacketTooShort); 
  headerLValue = this.data.extractBits(this.nextBitIndex, bitsToExtract); 
  headerLValue.valid$ = true;
	if headerLValue.isNext$ {
		verify(headerLValue.nextIndex$ < headerLValue.size, error.StackOutOfBounds);
		headerLValue.nextIndex$ = headerLValue.nextIndex$ + 1; 
  }
  this.nextBitIndex += bitsToExtract;
}
```

- **parse_source_routing**

```c
state parse_source_routing {
  packet.extract(hdr.source_routes.next);
  transition select(hdr.source_routes.last.last_header){
    1: parse_ipv4;
    default: parse_source_routing;
  }
}
```

在parse_source_routing状态，我们需要注意 `packet.extract(hdr.source_routes.next)`，这里out类型的变量是source_routes下一个位置。初始时，next 指向堆栈的第一个元素，当成功调用extract方法后，next将自动向前偏移，指向下一个元素。last指向 next 前面的那个元素（如果元素存在），即最近 extract 出来的那个元素。

```c
//初始：
                  packet.index
		       ↓
        packet: ETH |0 2 | 0 3 | 0 2 | 0 2 | 1 1 | IPV4 |

                      next
                       ↓
        hdr:	 ETH|	  |     |     |     |     |      |
  
//第一次执行完extract：
                       packet.index
			    ↓
        packet: ETH |0 2 | 0 3 | 0 2 | 0 2 | 1 1 | IPV4 |

                last  next
                  ↓    ↓
        hdr:	 ETH| 0 2 |     |     |     |     |     |
```

执行完extract后，如果当前hdr.source_routes.last.last_header仍为0，那么数据包的下一个状态仍为parse_source_routing。程序会一直持续该循环，直到当前hdr.source_routes.last.last_header为1，那么进入parse_ipv4状态。在parse_ipv4状态中，程序执行完一次extract之后，数据包报文头部的解析就结束了，之后程序进入到control流程，control部分的处理将会在source_routing.p4部分中分析。

- **Deparser**

```c
control MyDeparser(packet_out packet, in headers hdr) {
    apply {

        //parsed headers have to be added again into the packet.
        packet.emit(hdr.ethernet);
        packet.emit(hdr.source_routes);
        packet.emit(hdr.ipv4);

    }
}
```

在deparser阶段，control会将hdr（in类型）中的数据写入packet中（packet_out类型，与packet_in一样在core.p4中定义，但是其本身是out类型的数据）。这里需要注意的是`emit`函数，该函数同extract函数一样，需要我们仔细理解，emit函数的伪代码定义如下：

```c
//emi函数定义：
void emit<T>(T data) {
        if (isHeader(T))
            if(data.valid$) {
                this.data.append(data);
								this.lengthInBits += data.lengthInBits; 
            }
        else if (isHeaderStack(T)){
            for (e : data){
                 emit(e);
            }
        }
        else if (isHeaderUnion(T) || isStruct(T)){
            for (f : data.fields$){
                 emit(e.f)
            }
        }
        // Other cases for T are illegal
}
```

对于`packet.emit(hdr.ethernet)`语句，emit函数的输入变量hdr.ethernet属于header类型，调用函数后会直接进入第一个if分支，由于hdr.ethernet非空，同样满足第二个if分支判定，函数最终会在packet的首部直接填充该部分内容，并将packet的index指针移动到下一个位置。

而`packet.emit(hdr.source_routes)`语句比较复杂，我们需要注意的是hdr.source_routes是一个header stack（详情见header.p4中的定义`source_routing_t[MAX_HOPS] source_routes`），执行emit后程序将会进入第一个else if语句，然后循环遍历hdr.source_routes中的每一个元素hdr.source_routes[i]，并且以hdr.source_routes[i]为输入，递归调用emit函数。很显然hdr.source_routes[i]是header类型变量，但与hdr.ethernet不同的是，此时hdr.source_routes[i]中并没有任何内容（source_routing.p4中`control MyIngress{}`调用pop_front()函数去掉了hdr.source_routes中的内容，详情可以参考source_routing.p4部分的分析），并且在source_routing.p4的control流程中也没有调用`setvalid`函数使hdr.source_routes的有效位置为1，因此在递归调用emit函数过程中，程序进入第一个if分支判定后不满足第二个if分支判定的条件，最终不会往header中填充任何内容。

最后的`packet.emit(hdr.ipv4)`语句同`packet.emit(hdr.ethernet)`的处理流程一样，这里不再过多赘述。

---



##### (3) source_routing.p4

```c
#include <core.p4>
#include <v1model.p4>

//My includes
#include "include/headers.p4"
#include "include/parsers.p4"


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action set_normal_ethernet(){
        hdr.ethernet.etherType = TYPE_IPV4;
    }

    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {

        //set the src mac address as the previous dst, this is not correct right?
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;

       //set the destination mac address that we got from the match in the table
        hdr.ethernet.dstAddr = dstAddr;

        //set the output port that we also get from the table
        standard_metadata.egress_spec = port;

        //decrease ttl by 1
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;

    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table device_to_port {

        key = {
            hdr.source_routes[0].switch_id: exact;
        }

        actions = {
            ipv4_forward;
            NoAction;
        }
        size = 128;
        default_action = NoAction();

    }

    apply {

        //only if IPV4 the rule is applied. Therefore other packets will not be forwarded.
        if (hdr.source_routes[0].isValid() && device_to_port.apply().hit){
            //if it is the last header then.
            if (hdr.source_routes[0].last_header == 1 ){
               set_normal_ethernet();
            }
            hdr.source_routes.pop_front(1);
        }

        else if (hdr.ipv4.isValid()){
            ipv4_lpm.apply();
            //it means that it did not hit but that there is something to remove..
            if (hdr.source_routes[0].isValid()){
                //if it is the last header then.
                if (hdr.source_routes[0].last_header == 1 ){
                   set_normal_ethernet();
                }
                hdr.source_routes.pop_front(1);

            }
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {

    apply {}

}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	      hdr.ipv4.ihl,
              hdr.ipv4.dscp,
              hdr.ipv4.ecn,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}




/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

//switch architecture
V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
```

观察p4代码我们可以看到，这部分代码主要由4个control构成：`MyVerifyChecksum`、`MyIngress`、`MyEgress`、`MyComputeChecksum`。在这四个control中，` MyVerifyChecksum` 、`MyEgress`部分仅给出了定义，无任何apply实现，因此这几个control可以暂时忽略。剩下的两个conrtol中需要重点关注的是` MyIngress`，源路由p4程序中的大部分控制逻辑都是由`MyIngress`实现的。

- **control MyIngress**

```c
// MyIngress定义
control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
	...
  ...
}
```

同parser一样，在分析`control MyIngress`的具体实现前我们需要注意它的定义



```c
// action定义
action drop() {
  ...
}

action set_normal_ethernet(){
  ...
}

action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
	...
}
```

`control MyIngress`定义了三个action，action可以按照c语言中的函数进行理解，action可以在apply中通过匹配table触发，也可以通过在apply直接调用函数触发。



```c
// table定义
table ipv4_lpm {
  key = {
    hdr.ipv4.dstAddr: lpm;
  }
  actions = {
    ipv4_forward;
    drop;
    NoAction;
  }
  size = 1024;
  default_action = NoAction();
}

table device_to_port {

  key = {
    hdr.source_routes[0].switch_id: exact;
  }

  actions = {
    ipv4_forward;
    NoAction;
  }
  size = 128;
  default_action = NoAction();

}
```

`control MyIngress`定义了两个table，通过匹配key去触发相应的action，key以及action的参数等可在表项的配置文件中定义。



```c
// header对象的push和pop函数定义（伪代码）
void push_front(int count) {
for (int i = this.size-1; i >= 0; i -= 1) {
        if (i >= count) {
            this[i] = this[i-count];
        } else {
            this[i].setInvalid();
} }
this.nextIndex = this.nextIndex + count;
if (this.nextIndex > this.size) this.nextIndex = this.size;
// Note: this.last, this.next, and this.lastIndex adjust with this.nextIndex
}

void pop_front(int count) {
    for (int i = 0; i < this.size; i++) {
        if (i + count < this.size) {
            this[i] = this[i+count];
        } 
      	else {
            this[i].setInvalid();
					}
    }
    if (this.nextIndex >= count) {
        this.nextIndex = this.nextIndex - count;
    } 
  	else {
        this.nextIndex = 0;
    }
// Note: this.last, this.next, and this.lastIndex adjust with this.nextIndex
}
```
push与pop可以根据以上伪代码进行理解。