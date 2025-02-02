FROM nginx:latest

# 安装 inotify-tools
RUN apt-get update && apt-get install -y inotify-tools && apt-get clean

# 将监控脚本加入容器
COPY monitor.sh /usr/local/bin/monitor.sh
RUN chmod +x /usr/local/bin/monitor.sh

# 启动监控脚本和 Nginx
CMD ["/bin/bash", "-c", "/usr/local/bin/monitor.sh & nginx -g 'daemon off;'"]

