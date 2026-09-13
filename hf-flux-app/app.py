
import os
import time
import base64
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # 从 .env 文件读取配置

HF_TOKEN = os.getenv("HF_TOKEN", "")
# 走 fal-ai 第三方推理通道
MODEL = os.getenv("HF_MODEL", "XLabs-AI/flux-RealismLora")
PROVIDER = os.getenv("HF_PROVIDER", "fal-ai").strip()

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """接收提示词 -> 调用 HuggingFace Inference Providers -> 返回 base64 图像"""
    if not HF_TOKEN:
        return jsonify({"error": "服务端未配置 HF_TOKEN，请在 .env 文件中填入你的 HuggingFace Token"}), 500

    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "提示词不能为空"}), 400

    # 官方客户端会自动处理 LoRA 权重注入、参数翻译等细节
    client = InferenceClient(provider=PROVIDER, api_key=HF_TOKEN)

    started = time.time()
    try:
        image = client.text_to_image(
            prompt,
            model=MODEL,
            width=1024,
            height=1024,
        )
    except Exception as e:
        msg = str(e)
        print(f"[ERROR] 调用失败: {msg[:300]}")
        return jsonify({"error": f"调用失败：{msg[:400]}"}), 502

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    elapsed = round(time.time() - started, 2)

    # 同时保存到本地 generated/ 目录，防止页面刷新后图像丢失
    os.makedirs("generated", exist_ok=True)
    filename = time.strftime("flux_%Y%m%d_%H%M%S.png")
    with open(os.path.join("generated", filename), "wb") as f:
        f.write(buf.getvalue())

    print(f"[SUCCESS] 生成成功，耗时 {elapsed}s，模型: {MODEL}，通道: {PROVIDER}，提示词: {prompt[:60]}")
    print(f"[SAVED] 图像已保存到 generated/{filename}")
    return jsonify({
        "image": f"data:image/png;base64,{b64}",
        "model": MODEL,
        "provider": PROVIDER,
        "elapsed": elapsed,
        "saved": f"generated/{filename}",
    })


if __name__ == "__main__":
    print(f"当前模型: {MODEL} ｜ 推理通道: {PROVIDER}")
    app.run(host="127.0.0.1", port=5000, debug=False)
