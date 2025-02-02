from flask import Flask, request
import subprocess
import hmac
import hashlib

app = Flask(__name__)


GITHUB_SECRET = '123456789'
PROJECT_DIR = "/mnt"

def verify_signature(data, signature):
    computed_signature = 'sha256=' + hmac.new(GITHUB_SECRET.encode(), data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed_signature, signature)

@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    if request.method == 'GET':
        return "Webhook endpoint is working", 200

    print("Headers:", request.headers)
    print("Request data:", request.data)

    signature = request.headers.get('X-Hub-Signature-256')
    if signature is None or not verify_signature(request.data, signature):
        print("Invalid signature")
        return "Forbidden", 403

    event_type = request.headers.get('X-GitHub-Event')
    if event_type == 'ping':
        return "Ping event received", 200
    print("GitHub Event:", event_type)
    if event_type == 'push':
        try:
            print("Executing git pull command...")
            result = subprocess.run(["git", "pull"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
            print("Pull successful:", result.stdout)
            return f"Pull successful: {result.stdout}", 200
        except subprocess.CalledProcessError as e:
            print("Git pull failed:", e.stderr) 
            return f"Pull failed: {e.stderr}", 500
        except Exception as e:
            print("An unexpected error occurred:", str(e))  
            return f"An error occurred: {str(e)}", 500

    return "Event not handled", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
