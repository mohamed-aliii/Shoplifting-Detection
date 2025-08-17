from dotenv import load_dotenv
import os
import torch


load_dotenv()

APP_NAME = os.getenv('APP_NAME')
APP_VERSION = os.getenv('APP_VERSION')
APP_ENV = os.getenv('APP_ENV')
API_KEY = os.getenv('API_KEY')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifact", 'shoplifting_model.pth')

try:
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval() 
        print(f"Model loaded successfully ")
    else:
        print(f"Model file not found ")
        model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
