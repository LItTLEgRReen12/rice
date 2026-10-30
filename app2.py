import os
import io
import urllib.request
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from model import ComplexCNN  # 导入你的模型结构

# ================= 1. 自动下载模型逻辑 =================
# 填入你刚刚复制的 Release 下载链接
MODEL_URL = "https://github.com/LItTLEgRReen12/rice/releases/download/v1.0.0/best_rice_model.pth"
MODEL_PATH = "best_rice_model.pth"

if not os.path.exists(MODEL_PATH):
    print("云端未找到模型，正在从 GitHub Releases 下载...")
    # 下载模型文件
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("模型下载完成！")
else:
    print("模型文件已存在，跳过下载。")
# =======================================================

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. 加载模型 =================
device = torch.device("cpu")
print("正在加载模型到内存...")
# 加上 weights_only=False 避免 PyTorch 新版报错
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
print("模型加载成功，API 准备就绪！")
# ==============================================

class_names = [
    'Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy_Rice_Leaf',
    'Leaf_Blast', 'Leaf_scald', 'Narrow_Brown_Leaf_Spot',
    'Neck_Blast', 'Rice_Hispa', 'Sheath_Blight', 'Tungro'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 3. API 接口 =================
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        disease_name = class_names[pred.item()]
        return {
            "status": "success",
            "data": {
                "disease_name": disease_name,
                "confidence": f"{conf.item() * 100:.2f}%",
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"message": "水稻病害识别云端API已启动，正在运行"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# ==============================================
