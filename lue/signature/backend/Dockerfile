# 使用官方 Python 基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app
# 安裝必要的 Python 套件
RUN pip install flask

# 建立簽名存儲目錄
RUN mkdir -p /app/signatures

# 將簽名目錄設置為可讀寫
RUN chmod -R 777 /app/signatures

# 對外暴露伺服器埠號
EXPOSE 5000

# 啟動應用程式
CMD ["python", "server.py"]

