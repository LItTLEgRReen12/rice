# 1. 使用 Python 3.10 作为基础环境
FROM python:3.10-slim

# 2. 在云端创建一个叫 /app 的文件夹作为工作区
WORKDIR /app

# 3. 把依赖清单复制进去
COPY requirements.txt .

# 4. 按照清单，安装依赖包（这步需要几分钟，因为它要下载 PyTorch）
RUN pip install --no-cache-dir -r requirements.txt

# 5. 把你的所有代码和模型复制到云端 /app 目录下
COPY . .

# 6. 告诉云端，我的服务用 8000 端口
EXPOSE 8000

# 7. 启动命令：用 uvicorn 运行 app.py 里的 app
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8000"]