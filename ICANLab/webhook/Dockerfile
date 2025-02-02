FROM python:3.9-slim

# 安裝所需的系統依賴
RUN apt-get update && apt-get install -y git && apt-get clean

# 設置工作目錄
WORKDIR /app

# 複製應用代碼
COPY webhook_server.py /app/

# 安裝 Flask
RUN pip install flask
# 添加用戶和組
RUN groupadd -g 1000 ycair && \
    useradd -m -u 1000 -g ycair -s /bin/bash ycair

# 切換到新用戶
USER ycair

# 暴露服務端口
EXPOSE 5000

# 啟動應用
CMD ["python", "webhook_server.py"]

