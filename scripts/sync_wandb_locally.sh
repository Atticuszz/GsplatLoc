#!/bin/zsh
# 指定本地服务器路径
LOCAL_SERVER_URL="http://localhost:8080"

# 设置 wandb 项目和实体
PROJECT="GspaltLoc"
ENTITY="atticuszz"

# 检查本地服务器是否可访问
if ! curl --output /dev/null --silent --head --fail "$LOCAL_SERVER_URL"; then
    echo "Error: Cannot access the local server at $LOCAL_SERVER_URL"
    exit 1
fi

# 获取服务器上的目录列表
directories=$(curl -s "$LOCAL_SERVER_URL" | grep -oP '(?<=href=")[^"]*(?=/")')

# 遍历目录列表
for dir in $directories
do
    echo "Syncing directory: $dir"
    # 这里假设每个目录下有一个 wandb 文件夹
    wandb_url="$LOCAL_SERVER_URL/$dir/wandb"

    # 检查 wandb 目录是否存在
    if curl --output /dev/null --silent --head --fail "$wandb_url"; then
        # 创建一个临时目录来下载数据
        temp_dir=$(mktemp -d)

        # 下载 wandb 数据到临时目录
        curl -s "$wandb_url" -o "$temp_dir/wandb_data.zip"

        # 解压数据
        unzip -q "$temp_dir/wandb_data.zip" -d "$temp_dir"

        # 同步数据到 Weights & Biases
        wandb sync "$temp_dir" --project "$PROJECT" --entity "$ENTITY"

        # 清理临时目录
        rm -rf "$temp_dir"

        echo "Finished syncing $dir"
        echo "------------------------"
    else
        echo "No wandb directory found in $dir"
    fi
done

echo "All directories have been processed."
