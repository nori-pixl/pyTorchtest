import os
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# 読み書き・学習対象とするファイル名
TXT_FILE = "kioku.txt"

# ファイルがなければ初期状態で自動作成
if not os.path.exists(TXT_FILE):
    with open(TXT_FILE, "w", encoding="utf-8") as f:
        f.write("AIに覚えさせたい言葉をここに書きます。\n")

# ----------------------------------------
# 1. PyTorch テキスト記憶用ネットワーク（RNN）の定義
# ----------------------------------------
class RNNMemoryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNMemoryModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        logits = self.fc(output)
        return logits

# ----------------------------------------
# 2. Web画面のデザイン（HTML / CSS）
# ----------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyTorch 記憶学習アプリ</title>
    
</head>
<body>
<div class="container">
    <h1>pyaitest</h1>
    
    {% if status_msg %}
        <div class="status">{{ status_msg }}</div>
    {% endif %}

    <!-- 1. kioku.txtの編集フォーム -->
    <div class="box">
        <h2></h2>
        <form action="/save" method="POST">
            <textarea name="content" placeholder="AIに覚えさせたい文章を入力してください...">{{ file_content }}</textarea>
            <button type="submit" class="btn-add">kioku.txt に上書き保存</button>
        </form>
    </div>

    <!-- 2. PyTorch学習トリガーボタン -->
    <div class="box">
        <h2</h2>
        <form action="/train_ai" method="POST">
            <button type="submit" class="btn-train">PyTorchに kioku.txt をすべて覚えさせる</button>
        </form>
    </div>
</div>
</body>
</html>
"""

# アプリケーションの状態保持用
status_msg = ""

# ----------------------------------------
# 3. Webサーバーのルート処理（Flask）
# ----------------------------------------
@app.route("/", methods=["GET"])
def index():
    global status_msg
    file_content = ""
    if os.path.exists(TXT_FILE):
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            file_content = f.read()
    
    msg = status_msg
    status_msg = "" 
    return render_template_string(HTML_TEMPLATE, file_content=file_content, status_msg=msg)

@app.route("/save", methods=["POST"])
def save_file():
    global status_msg
    content = request.form.get("content", "")
    with open(TXT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    status_msg =  kioku.txt を更新しました。"
    return redirect(url_for("index"))

@app.route("/train_ai", methods=["POST"])
def train_ai():
    global status_msg
    
    if not os.path.exists(TXT_FILE):
        status_msg = "kioku.txt が見つかりません。"
        return redirect(url_for("index"))
        
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        text_data = f.read().strip()
        
    if len(text_data) < 2:
        status_msg = "覚えさせる文字数が足りません（2文字以上必要です）。"
        return redirect(url_for("index"))

    try:
        # 1. テキストを文字単位で数値（ID）にマッピング
        chars = sorted(list(set(text_data)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        
        input_ids = [char_to_idx[ch] for ch in text_data]
        
        # 入力データ(X)と教師データ(Y)のテンソルを作成
        X = torch.tensor(input_ids[:-1]).unsqueeze(0) 
        Y = torch.tensor(input_ids[1:]).unsqueeze(0)  

        # 2. モデルと最適化の構築
        model = RNNMemoryModel(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        # 3. PyTorchでの学習ループ（記憶の叩き込み）
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X) 
            loss = criterion(output.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            optimizer.step()

        status_msg = f"PyTorchが kioku.txt（全{len(text_data)}文字）をインプットとして完全に学習しました！"

    except Exception as e:
        status_msg = f"❌ AIの学習中にエラーが発生しました: {str(e)}"
            
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
