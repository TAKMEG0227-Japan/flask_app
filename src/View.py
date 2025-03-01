# coding: utf-8

from flask import Flask, render_template

# Flask をインスタンス化
app = Flask(__name__)

# --- View側の設定 ---
# ルートディレクトリにアクセスした場合の挙動
@app.route('/')
# def以下がアクセス後の操作
def index():
  #  return 'Hello World!'
    return render_template('index.html')

import os

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
