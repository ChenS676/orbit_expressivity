#!/bin/bash

# 遍历当前目录下所有以 " copy.py" 结尾的 .py 文件
find . -type f -name '* copy.py' | while read oldname
do
    # 生成新的文件名（去掉" copy"）
    newname="$(echo "$oldname" | sed 's/ copy\.py$/.py/')"
    
    # 如果新文件已存在，避免覆盖
    if [ -e "$newname" ]; then
        echo "Skip: $newname already exists."
    else
        mv "$oldname" "$newname"
        echo "Renamed: $oldname --> $newname"
    fi
done