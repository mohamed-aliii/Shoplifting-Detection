from src.preprocessing import preprocess_video
import torch
import torch.nn as nn
import torchvision.models.video as video_models

def sota_model(num_classes=1, pretrained=False):
    model = video_models.mc3_18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(128, num_classes)
    )
    return model

def run_inference(model, video_path, num_frames=16, device="cuda", threshold=0.5):
    """
    Runs inference on a single video file.
    """
    # Preprocess
    input_tensor = preprocess_video(video_path, num_frames, device)
    input_tensor = input_tensor.permute(0, 2, 1, 3, 4).to(device)
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # If binary classification
    if output.shape[1] == 1:
        probs = torch.sigmoid(output).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)
    else:
        probs = torch.softmax(output, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    return preds, probs