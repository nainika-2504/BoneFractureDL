
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Bone Fracture Detection API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 2)
    )
    model.load_state_dict(torch.load(
        "/content/best_bone_model_v2.pth",
        map_location=device
    ))
    model.to(device)
    model.eval()
    return model

model = load_model()
class_names = ["fracture", "normal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "Bone Fracture Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ResNet50", "classes": class_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Accept any image type by checking filename extension
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {filename}"}
        )

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = outputs.argmax(1).item()
        conf    = probs[0][pred].item() * 100

    return JSONResponse(content={
        "prediction" : class_names[pred],
        "confidence" : round(conf, 2),
        "fracture"   : round(probs[0][0].item() * 100, 2),
        "normal"     : round(probs[0][1].item() * 100, 2),
        "status"     : "success"
    })
