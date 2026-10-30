import os
import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from model import ComplexCNN  # 这里导入你刚才建好的模型文件

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 加载模型
device = torch.device("cpu")
# 加上 weights_only=False 避免 PyTorch 新版报错
model = torch.load("best_rice_model.pth", map_location=device, weights_only=False)
model.eval()

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

# 2. 接口
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
    return {"message": "水稻病害识别云端API已启动"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)