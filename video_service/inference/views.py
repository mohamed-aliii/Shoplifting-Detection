from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .models import VideoUpload
import torch
import sys
import os
from pathlib import Path
import torch.nn as nn
import torchvision.models.video as video_models
# Add the parent directory to system path to import src modules
root_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(root_dir)
from src.inference import sota_model
from src.inference import run_inference

model = None

def load_model():
    global model
    if model is None:
        try:
            # Create the same model architecture used in training
            model = sota_model(num_classes=1).to(settings.DEVICE)
            # Load the saved state dict
            state_dict = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    return model

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file uploaded'}, status=400)
    
    video_file = request.FILES['video']
    video_upload = VideoUpload.objects.create(video=video_file)
    
    try:
        # Get model
        model = load_model()
        
        try:
            # Run inference using your inference function
            predictions, probabilities = run_inference(
                model=model,
                video_path=video_upload.video.path,
                num_frames=settings.NUM_FRAMES,
                device=settings.DEVICE,
                threshold=settings.DETECTION_THRESHOLD
            )
            
            # Update model with results
            video_upload.processed = True
            video_upload.is_shoplifting = bool(predictions[0])  # First prediction
            video_upload.confidence = float(probabilities[0])  # First probability
            video_upload.save()
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'id': video_upload.id,
                    'is_shoplifting': video_upload.is_shoplifting,
                    'confidence': video_upload.confidence
                })
            else:
                result_message = 'Shoplifting detected!' if video_upload.is_shoplifting else 'No shoplifting detected.'
                messages.info(request, f'{result_message} (Confidence: {video_upload.confidence:.2%})')
                return redirect('video_list')
            
        except Exception as e:
            video_upload.delete()  # Clean up on error
            raise e
            
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'error': str(e)}, status=500)
        else:
            messages.error(request, f'Error processing video: {str(e)}')
            return redirect('home')

def home(request):
    return render(request, 'inference/home.html')

def video_list(request):
    videos = VideoUpload.objects.all().order_by('-uploaded_at')
    return render(request, 'inference/video_list.html', {'videos': videos})
