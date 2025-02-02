from flask import Flask, request, jsonify
import base64
import os
from datetime import datetime

app = Flask(__name__)

# 設定簽名儲存目錄
UPLOAD_FOLDER = 'signatures'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload-signature', methods=['POST'])
def upload_signature():
    try:
        # 從請求中獲取簽名資料
        data = request.json
        if 'signature' not in data:
            return jsonify({'error': 'No signature data found'}), 400

        signature_data = data['signature']

        # 解碼 Base64 圖片資料
        if signature_data.startswith('data:image/png;base64,'):
            signature_data = signature_data.replace('data:image/png;base64,', '')

        decoded_image = base64.b64decode(signature_data)

        # 生成檔案名稱 (使用當前時間避免重名)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'signature_{timestamp}.png'
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # 將圖片儲存到伺服器目錄
        with open(file_path, 'wb') as file:
            file.write(decoded_image)

        return jsonify({'message': 'Signature uploaded successfully', 'file_path': file_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

