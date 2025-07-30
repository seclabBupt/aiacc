# `.sh`文件

`.sh`文件是 **Shell** 脚本文件，是包含一系列命令的纯文本文件，由 Unix/Linux Shell（命令行解释器）读取并执行。\
**核心作用**： 将需要在命令行手动输入的一系列命令**自动化**，组合成一个可执行的程序。\
**关键特点**

1. **纯文本：** 可以用任何文本编辑器（如 `vim`, `nano`, `VSCode`, `Notepad++`）创建和编辑。
2. **首行 Shebang (`#!`):**
    * 文件的第一行通常是 `#!/bin/bash` (最常见) 或 `#!/bin/sh`。
    * 这告诉系统使用哪个 Shell 程序来执行这个脚本（例如 `/bin/bash` 或 `/bin/sh`）。
3. **需要执行权限：**
    * 创建后，需要通过 `chmod +x yourscript.sh` 命令赋予它可执行权限，才能像程序一样运行 `./yourscript.sh`。
    * 也可以直接通过指定解释器运行：`bash yourscript.sh` 或 `sh yourscript.sh`（此时即使没有 `+x` 权限也能运行，但更推荐赋予权限的方式）。
4. **包含 Shell 命令：**
    * 脚本里可以写任何能在终端直接运行的命令（`ls`, `cp`, `mv`, `grep`, `awk`, `sed` 等）。
    * 还支持 Shell 编程特性：变量、条件判断 (`if..then..else`)、循环 (`for`, `while`)、函数、接受命令行参数 (`$1`, `$2`, `$@`)、读取用户输入等。
5. **平台：**
    * 主要在 **Linux、macOS 和其他 Unix-like 系统** 上原生运行。
    * 在 Windows 上可以通过 **WSL (Windows Subsystem for Linux)、Cygwin、Git Bash、MinGW/MSYS2** 等兼容环境运行。

以下是 Shell 脚本（特别是 Bash，作为最常用的 Shell）中最核心、最常用的语法概念和结构，掌握这些就能编写实用的脚本：

## **1. 变量 (Variables)**

* **定义与赋值：** `变量名=值` **(注意：等号两边不能有空格！)**

    ```bash
    name="Alice"
    count=10
    files=$(ls)  # 将命令输出赋值给变量
    today=$(date +%Y-%m-%d)
    ```

* **使用变量：** `$变量名` 或 `${变量名}` (后者更清晰，尤其在变量名后紧接其他字符时)

    ```bash
    echo "Hello, $name"
    echo "Count: ${count}th item"
    echo "Today is $today"
    echo "Files: $files"
    ```

* **特殊变量：**

  * `$0`：脚本名称
  * `$1, $2, ... $9`：脚本的第 1 到第 9 个参数
  * `$#`：传递给脚本的参数个数
  * `$@`：所有位置参数的列表 (每个参数独立)
  * `$*`：所有位置参数的列表 (作为一个字符串)
  * `$?`：上一个命令的退出状态 (0 通常表示成功，非 0 表示失败)
  * `$$`：当前 Shell 进程的 PID

## **2. 引号 (Quotes)**

* **单引号 `''`：** 内部内容原样输出，不进行变量替换或命令替换。

    ```bash
    echo '$name will be printed literally' # 输出: $name will be printed literally
    ```

* **双引号 `""`：** 内部内容会进行变量替换 (`$var`) 和命令替换 (`$(cmd)` 或 `` `cmd` ``)。

    ```bash
    echo "Hello, $name" # 输出: Hello, Alice (假设 name="Alice")
    echo "Today: $(date)"
    ```

* **反引号 `` ` ` ``：** 等同于 `$()`，用于命令替换（推荐使用 `$()`，更清晰且可嵌套）。

    ```bash
    old_way=`ls`
    new_way=$(ls)
    ```

## **3. 条件判断 (Conditionals)**

* **`test` 命令 或 `[ ]` / `[[ ]]`：** 用于检查条件。
  * `test -f file.txt` 或 `[ -f file.txt ]`：检查文件是否存在且是普通文件。
  * `[[ -d /path/to/dir ]]`：检查目录是否存在 (推荐 `[[ ]]`，功能更强大，支持 `&&`, `||`, 模式匹配)。
* **常用文件/字符串测试符:**
  * `-e file`：文件/目录是否存在
  * `-f file`：文件是否存在且是普通文件
  * `-d dir`：目录是否存在
  * `-r/w/x file`：文件是否可读/写/执行
  * `-s file`：文件存在且大小 > 0
  * `-z "$str"`：字符串长度是否为 0 (空)
  * `-n "$str"`：字符串长度是否不为 0 (非空)
  * `"$str1" = "$str2"` / `"$str1" == "$str2"`：字符串相等
  * `"$str1" != "$str2"`：字符串不相等
  * `$num1 -eq $num2`：数字相等 (equal)
  * `$num1 -ne $num2`：数字不相等 (not equal)
  * `$num1 -lt $num2`：小于 (less than)
  * `$num1 -le $num2`：小于等于 (less or equal)
  * `$num1 -gt $num2`：大于 (greater than)
  * `$num1 -ge $num2`：大于等于 (greater or equal)
* **`if...then...elif...else...fi` 语句：**

    ```bash
    if [[ -f "$filename" ]]; then
        echo "File $filename exists."
    elif [[ -d "$filename" ]]; then
        echo "$filename is a directory."
    else
        echo "$filename does not exist or is not a regular file/dir."
    fi

    # 检查命令是否成功
    if grep -q "pattern" file.txt; then
        echo "Pattern found."
    else
        echo "Pattern not found."
    fi

    # 检查变量值
    if [[ "$count" -gt 10 ]]; then
        echo "Count is greater than 10."
    fi
    ```

## **4. 循环 (Loops)**

* **`for` 循环：** 遍历列表或命令输出。

    ```bash
    # 遍历单词列表
    for fruit in apple banana orange; do
        echo "I like $fruit"
    done

    # 遍历当前目录下的所有 .txt 文件
    for file in *.txt; do
        echo "Processing $file..."
        # 对文件进行操作 (e.g., cp "$file" "backup_$file")
    done

    # 遍历命令输出 (每行一个元素)
    for user in $(cut -d: -f1 /etc/passwd | head -n 5); do
        echo "User: $user"
    done

    # C 风格 for 循环
    for ((i=1; i<=5; i++)); do
        echo "Iteration $i"
    done
    ```

* **`while` 循环：** 当条件为真时执行循环体。

    ```bash
    count=1
    while [[ $count -le 5 ]]; do
        echo "Count: $count"
        ((count++))  # 递增 count，等同于 count=$((count + 1))
    done

    # 逐行读取文件
    while IFS= read -r line; do
        echo "Line: $line"
    done < "input.txt"  # 输入重定向

    # 无限循环 (通常搭配 break)
    while true; do
        # 做某些事情...
        if [[ "$condition" == "met" ]]; then
            break  # 跳出循环
        fi
        sleep 1
    done
    ```

* **`until` 循环：** 当条件为假时执行循环体（与 `while` 相反）。

    ```bash
    count=1
    until [[ $count -gt 5 ]]; do
        echo "Count: $count"
        ((count++))
    done
    ```

## **5. 函数 (Functions)**

* **定义：**

    ```bash
    function myfunc() {
        # 函数体
        echo "This is my function"
        local local_var="I'm local" # 定义局部变量 (只在函数内可见)
        return 0 # 返回值 (0-255，通常 0 表示成功)
    }

    # 另一种定义方式 (更符合 POSIX)
    my_other_func() {
        echo "Another function"
    }
    ```

* **调用：** 直接写函数名，如同命令。

    ```bash
    myfunc
    my_other_func
    ```

* **参数：** 函数内部使用 `$1`, `$2`, ... `$#`, `$@` 等访问传递给它的参数。

    ```bash
    greet() {
        echo "Hello, $1!"
    }
    greet "Bob" # 输出: Hello, Bob!
    ```

## **6. 输入/输出重定向 (Input/Output Redirection)**

* `command > file`：将命令的标准输出重定向到文件（覆盖）。
* `command >> file`：将命令的标准输出重定向到文件（追加）。
* `command < file`：将文件内容作为命令的标准输入。
* `command 2> error.log`：将命令的标准错误重定向到文件（覆盖）。
* `command 2>> error.log`：将命令的标准错误重定向到文件（追加）。
* `command &> output.log` 或 `command > output.log 2>&1`：将标准输出和标准错误都重定向到同一个文件（覆盖）。
* `command >> output.log 2>&1`：将标准输出和标准错误都重定向到同一个文件（追加）。
* `command1 | command2`：管道，将 `command1` 的标准输出作为 `command2` 的标准输入。

## **7. 命令替换 (Command Substitution)**

* 将命令的输出结果赋值给变量或嵌入到其他命令中。

    ```bash
    file_count=$(ls | wc -l) # 计算当前目录文件数
    echo "There are $file_count files."

    echo "The date is $(date)"
    ```

## **8. 算术运算 (Arithmetic Expansion)**

* 使用 `$(( ))` 进行整数算术运算。

    ```bash
    sum=$((5 + 3))
    count=$((count + 1))
    product=$(( $num1 * $num2 ))
    remainder=$(( $dividend % $divisor ))
    ```

## **9. 退出脚本 (Exiting)**

* `exit [n]`：立即退出脚本，并返回状态码 `n` (0 表示成功，非 0 表示错误)。如果省略 `n`，则退出状态为最后一条命令的退出状态。

## **10. 注释 (Comments)**

* 以 `#` 开头的行是注释，解释器会忽略。

    ```bash
    # 这是一个单行注释

    : '
    这是一个
    多行注释
    (Bash 特有方式)
    '
    ```

## **11. 读取用户输入 (Reading Input)**

* `read [-p prompt] variable`：从标准输入读取一行，存入变量。

    ```bash
    read -p "Enter your name: " username
    echo "Hello, $username!"
    ```

## **12. 处理选项和参数 (getopts)**

* 用于解析命令行选项（如 `-f`, `--file`）。

    ```bash
    while getopts ":a:b:" opt; do # a 和 b 需要参数 (后面有冒号:)
      case $opt in
        a)
          arg_a="$OPTARG"
          ;;
        b)
          arg_b="$OPTARG"
          ;;
        \?)
          echo "Invalid option: -$OPTARG" >&2
          exit 1
          ;;
        :)
          echo "Option -$OPTARG requires an argument." >&2
          exit 1
          ;;
      esac
    done
    shift $((OPTIND - 1)) # 移除已处理的选项，剩下的是位置参数
    # 现在可以处理 $arg_a, $arg_b 和位置参数 $1, $2...
    ```

## **总结关键点：**

1. **变量赋值无空格：** `var=value`。
2. **变量引用加 `$`：** `echo $var` 或 `echo ${var}`。
3. **条件判断用 `[ ]` 或 `[[ ]]`：** 注意内部空格 `[ -f "$file" ]`。
4. **字符串比较用 `=` / `==` / `!=`：** 在 `[[ ]]` 内。
5. **数字比较用 `-eq`, `-ne`, `-lt`, `-le`, `-gt`, `-ge`：** 在 `[ ]` 或 `[[ ]]` 内。
6. **循环和条件语句块以 `do...done`, `then...fi` 结束。**
7. **函数参数通过 `$1`, `$2`... 访问。**
8. **命令输出赋值用 `$(command)`。**
9. **算术运算用 `$((expression))`。**
10. **善用引号处理包含空格或特殊字符的字符串/文件名。**
11. **`exit` 控制脚本退出状态。**
