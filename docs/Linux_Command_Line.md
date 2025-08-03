# Linux系统(服务器)常用指令

## 最常用

### ls (列出文件)

- `ls -la` 给出当前目录下所有文件的一个长列表，包括以句点开头的“隐藏”文件
- `ls a*` 列出当前目录下以字母 a 开头的所有文件
- `ls -l *.doc` 给出当前目录下以.doc 结尾的所有文件

### mv (移动和重命名文件)

- `mv oldfile newfile` 将 oldfile 重命名为 newfile
- `mv oldfile /tmp` 把当前目录下的 oldfile 移动到/tmp/目录下

### rm (删除文件和目录)

- `rm oldfile` 删除文件 oldfile
- `rm *` 删除当前目录下的所有文件（未隐藏文件）。rm 命令不删除目录，除非也指定了-r(递归)参数。
- `rm -rf domed` 删除 domed 目录以及它所包含的所有内容
- `rm -i a*` 删除当前目录下所有以字母 a 开头的文件，并且在每次删除时，提示用户进行确认

### cd (更改目录)

- `cd ~` 切换到主目录
- `cd /tmp` 切换到目录/tmp
- `cd dir` 切换到当前目录下的 dir 目录
- `cd /` 切换到根目录
- `cd ..` 切换到到上一级目录
- `cd ../../` 切换到上二级目录
- `cd ~` 切换到用户目录，比如是 root 用户，则切换到/root 下

### mkdir (建立目录)

- `mkdir photos` 在当前目录中建立名为 photos 的目录
- `mkdir -p this/that/photos` 在当前目录下建立指定的嵌套子目录

### exit（退出服务器）

- `exit`、`logout`断开链接connection closed/disconnected
- 如果你有多个终端会话打开，你可以使用命`exit all`来关闭所有的终端会话并退出服务器。
- 在Xshell的命令行界面，按下`Ctrl`+`D`组合键也可以退出当前会话。
- 如果当前正在运行一个命令或程序，可以按下`Ctrl`+`C`组合键中断操作，然后再使用exit或logout命令退出会话。

>需要注意的是，以上方法只是退出当前的终端会话，不会关闭Linux服务器。如果要断开与服务器的连接，可以在Xshell中选择关闭会话或断开连接的选项。

### cp (复制文件)

- `cp oldfile newfile.bak` 把文件复制为新文件 newfile.bak
- `cp oldfile /home/data/` 把文件 oldfile 从当前目录复制到/home/data/目录下
- `cp * /tmp` 把当前目录下的所有未隐藏文件复制到/tmp/目录下
- `cp -a docs docs.bak` 递归性地把当前目录下的 docs 目录复制为新目录 docs.bak,保持文件属性，并复制所有的文件，包括以句点开头的隐藏文件。为了方便起见，-a 选项包含-R 选项
- `cp -i` 在覆盖前询问用户
- `cp -v` 告诉用户正在做什么

### chmod（脚本执行）

详情见[Linux chmod命令](https://www.runoob.com/linux/linux-comm-chmod.html)

> `chmod [选项] 权限模式 文件...`\
> `chmod [选项] --reference=参考文件 文件...`

常用选项\
`-c` : 若该文件权限确实已经更改，才显示其更改动作\
`-f` : 若该文件权限无法被更改也不要显示错误讯息\
`-v` : 显示权限变更的详细资料\
`-R` : 对目前目录下的所有文件与子目录进行相同的权限变更(即以递归的方式逐个变更)\
`--help` : 显示辅助说明\
`--version` : 显示版本\

>`[ugoa...][ [+-=] [rwxX]...][,...]`

其中：\
`u`表示该文件的拥有者，`g`表示与该文件的拥有者属于同一个群体(group)者，`o`表示其他以外的人，`a`表示这三者皆是。\
`+`表示增加权限、`-`表示取消权限、`=`表示唯一设定权限。\
`r`表示可读取，`w`表示可写入，`x`表示可执行，`X`表示只有当该文件是个子目录或者该文件已经被设定过为可执行。

## **相关指令具体解释**

[Linux chmod 命令详解：权限管理的核心工具](Linux_Command_Line_chmod.md)\
[在 Linux 系统中创建新文件夹的完整指南](Linux_Command_Line_mkdir.md)\
[Linux 系统操作全面指南](Linux_Command_Line_guide.md)

## 不常用

### more、less (查看文件内容)

- `more /etc/passwd` 查看/etc/passwd 的内容
功能：分页显示命令
- `more file`
more 命令也可以通过管道符(|)与其他的命令一起使用，例如：

- `ps ux|more`
- `ls|more`
- `less /etc/passwd` 查看/etc/passwd 的内容

### grep (搜索文件内容)

- `grep bible /etc/exports` 在文件 exports 中查找包含 bible 的所有行
- `tail -100 /var/log/apache/access.log|grep 404` 在 WEB 服务器日志文件 access.log 的后 100 行中查找包含“404”的行
- `tail -100 /var/log/apache/access.log|grep -v googlebot` 在 WEB 服务器日志文件 access.log 的后 100 行中，查找没有被 google 访问的行
- `grep -v ^# /etc/apache2/httpd.conf` 在主 apache 配置文件中，查找所有非注释行 (10)命令 find——查找文件
- `find .-name *.rpm` 在当前目录中查找 rpm 包
- `find .|grep page` 在当前目录及其子目录中查找文件名包含 page 的文件 locate traceroute 在系统的任何地方查找文件名包含 traceroute 的文件

### vi (编辑文件)

- `vi /etc/bubby.txt` 用 vi 编辑文件/etc/bubby.txt
- `vim /etc/bubby.txt` 用 vi 编辑文件/etc/bubby.txt
  - i 插入
  - esc 退出编辑
  - :wq 保存并退出
  - :q 在文件未做任何修改的情况下退出
  - :w 保存文件
  - 复制： 在退出编辑的状态下把光标移动到某一行，输入行数，按下yy
  - 粘贴： 光标移动到要粘贴的行，按下p
  - ：q! 强制退出，不保存对文件所作的修改

### rz、sz (文件上传下载)

- `sudo rz` 即是接收文件，xshell 就会弹出文件选择对话框，选好文件之后关闭对话框，文件就会上传到 linux 里的当前目录 。
- `sudo sz file.xxx` 就是发文件到 windows 上（保存的目录是可以配置）

### cat (显示文件内容)

- `cat file.xxx`

### ps (查看进程)

- `ps [options]`

DESCRIPTION（描述）：ps 命令显示运行程序选项的一些信息。如果你想显示选项的一些重复信息，请使用 top 命令替代。 用标准语法查看系统上的每一个进程。

- `ps -e`
- `ps -ef`
- `ps -eF`
- `ps -ely`

### kill (杀掉进程)

- `kill -signal %jobnumber`
- `kill -l` 参数： -l ：这个是 L 的小写，列出目前 kill 能够使用的讯号 (signal) 有哪些
signal ：代表给予后面接的那个工作什么样的指示啰！用 man 7 signal 可知：

- `-1` ：重新读取一次参数的设定档 (类似 reload)；
- `-2` ：代表与由键盘输入 [ctrl]-c 同样的动作；
- `-9` ：立刻强制删除一个工作；
- `-15`：以正常的程序方式终止一项工作。与 -9 是不一样的。

范例一：找出目前的 bash 环境下的背景工作，并将该工作删除。

[root@linux ~]# jobs

[1]+ Stopped vim bashrc [root@linux ~]# kill -9 %1

[1]+ 已砍掉 vim bashrc (16)命令stop、start——重启tomcat ./catalina.sh stop

./catalina.sh start

### top (查看 cpu、内存)

### pwd (查看当前路径)

### tar (打包、解包 rar)

- `tar -cvf **.tar a.jsp b.java` 将 a 和 b 打成 tar 包
- `tar -xvf **.tar` 将**.tar 解包

### tail (查看文件详细信息)

- `tail -f aaa.txt`
- `tail -n x aaa.log` 看 aaa.txt 文件的详细信息 x:最后几行

### head (查看文件的名字和后缀)

- `head -n x aaa.log` x:开始几行 aaa.log：要查看的文件的名字和后缀

### diff (比较文件内容)

- `diff dir1 dir2` 比较目录 1 与目录 2 的文件列表是否相同，但不比较文件的实际内容，不同则列出
- `diff file1 file2` 比较文件 1 与文件 2 的内容是否相同，如果是文本格式的文件，则将不相同的内容显示，如果是二进制代码则只表示两个文件是不同的
- `comm file1 file2` 比较文件，显示两个文件不相同的内容

### ln (建立连接)

- `ln source_path target_path` 硬连接
- `ln -s source_path target_path` 软连接

### touch (创建一个空文件)

- `touch aaa.txt` 创建一个空文件，文件名为 aaa.txt

### man (看某个命令的帮助)

- `man ls` 显示 ls 命令的帮助内容

### w (显示登录用户的详细信息)

- `w`

### who (显示登录用户)

- `who`

### last (查看最近那些用户登录系统)

- `last`

### date (系统日期设定)

- `date -s “060520 06:00:00″` 设置系统时期为 2006 年 5 月 20 日 6 点整。

### clock (时钟设置)

- `clock –r` 对系统 Bios 中读取时间参数
- `clock –w` 将系统时间(如由 date 设置的时间)写入 Bios

### uname (查看系统版本)

- `uname -R` 显示操作系统内核的 version

### reboot、shutdown (关闭和重新启动计算机)

- `reboot` 重新启动计算机
- `shutdown -r now` 重新启动计算机，停止服务后重新启动计算机
- `shutdown -h now` 关闭计算机，停止服务后再关闭系统
- `halt` 关闭计算机
一般用 shutdown -r now,在重启系统是，关闭相关服务，shutdown -h now 也是如此。

### su (切换用户)

- `su -` 切换到 root 用户
- `su – pyl` 切换到 pyl 用户，

### free (查看内存和 swap 分区使用情况)

- `free -tm`

### uptime (现在的时间 ，系统开机运转到现在经过的时间，连线的使用者数量，最近一分钟，五分钟和十五分钟的系统负载)

- `uptime`

### vmstat (监视虚拟内存使用情况)

- `vmstat`

### iostat (磁盘吞吐量)

- `-c` 只显示 CPU 行
- `-d` 显示磁盘行
- `-k` 以千字节为单位显示磁盘输出
- `-t` 在输出中包括时间戳
- `-x` 在输出中包括扩展的磁盘指标

### clear (清屏)

- `clear`

### tomcat (重启)

- `tomcat`
