# <div align="center"> Shoplifting Detection System </div>

A deep learning-based video analysis system that detects shoplifting incidents using a Mixed 2D/3D CNN architecture. The system is implemented as a Django web service that processes video uploads and provides real-time detection results.

## Project Overview

This project implements a shoplifting detection system using deep learning techniques. It combines:
- A PyTorch-based video classification model (mc3_18)
- A Django web interface for video upload and analysis
- Real-time video processing pipeline
- Results visualization and storage

## Project Structure

```
├── Shop DataSet/
│   ├── non shop lifters/      # Non-shoplifting video samples
│   └── shop lifters/          # Shoplifting video samples
├── video_service/             # Django web application
│   ├── inference/             # Main application
│   │   ├── templates/         # HTML templates
│   │   │   ├── home.html     # Upload interface
│   │   │   └── video_list.html# Results view
│   │   ├── views.py          # View controllers
│   │   └── models.py         # Database models
│   ├── src/                   # Core functionality
│   │   ├── inference.py      # Model inference
│   │   └── preprocessing.py   # Video preprocessing
│   └── media/                 # Uploaded videos storage
└── notebooks/                 # Jupyter notebooks
    └── modelling_torch.ipynb  # Model training notebook

```

## Features

- **Video Upload**: Support for various video formats through a user-friendly web interface
- **Real-time Analysis**: Asynchronous video processing with progress feedback
- **Results Dashboard**: View and manage analyzed videos with confidence scores
- **Video Preprocessing**: Automated frame extraction and normalization
- **Persistent Storage**: Database storage for video metadata and analysis results

## Technical Stack

- **Backend Framework**: Django
- **Deep Learning**: PyTorch
- **Model Architecture**: mc3_18 for video classification
- **Frontend**: Bootstrap, jQuery for async requests
- **Video Processing**: torchvision.io
- **Database**: SQLite3

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd shoplifting-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
cd video_service
python manage.py migrate
```

5. Configure model path in settings.py:
```python
MODEL_PATH = "path/to/your/shoplifting_model.pth"
```

## Usage

1. Start the Django development server:
```bash
python manage.py runserver
```

2. Access the web interface at `http://localhost:8000`

3. Upload a video for analysis:
   - Click "Choose File" to select a video
   - Click "Analyze Video" to start processing
   - Wait for the analysis results

4. View results:
   - Check the immediate analysis results on the upload page
   - Visit the "View All Videos" page to see historical results

## Model Details

- **Architecture**: R2Plus1D_18 (CNN-LSTM hybrid)
- **Input**: 16-frame video segments at 112x112 resolution
- **Output**: Binary classification (shoplifting/non-shoplifting)
- **Performance Metrics**:
  - Detection Threshold: 0.5
  - GPU acceleration supported

## API Endpoints

- `/`: Home page with upload form
- `/predict/`: Video analysis endpoint (POST)
- `/video-list/`: Historical results view

## Configuration

Key settings in `video_service/settings.py`:
```python
MODEL_PATH = "path/to/model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
DETECTION_THRESHOLD = 0.5
```

## Development

- **Model Training**: Refer to `notebooks/modelling_torch.ipynb`
- **Custom Preprocessing**: Modify `src/preprocessing.py`
- **Inference Pipeline**: Adjust `src/inference.py`

## Error Handling

The system includes comprehensive error handling for:
- Invalid video formats
- Processing failures
- Model loading issues
- Resource unavailability

## Performance Considerations

- GPU acceleration recommended for faster inference
- Video preprocessing optimized for memory efficiency
- Async processing prevents UI blocking

## Security Features

- CSRF protection enabled
- File upload validation
- Secure media storage
- Django security middleware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2025 Mohamed Ali Ghonem


## Acknowledgments

- Dataset sources
- PyTorch team
- Django community


