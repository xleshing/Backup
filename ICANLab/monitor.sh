#!/bin/bash

# 监控 /usr/share/nginx/html 目录的文件变更
inotifywait -m /usr/share/nginx/html -e create -e modify -e delete |
while read path action file; do
    echo "$(date): Detected $action on $file. Reloading Nginx..."
    nginx -s reload
done

