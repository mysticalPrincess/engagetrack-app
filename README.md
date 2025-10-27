# EngageTrack - Real-Time Student Engagement Detection

A Flask-based web application that uses YOLOv11 for real-time student engagement detection with PostgreSQL database integration.

## Features

- 🎥 **Live Video Detection**: Real-time engagement analysis with bounding boxes
- 👥 **Student Tracking**: Consistent ID assignment for individual student monitoring
- 🚨 **Smart Alerts**: Automatic red alerts for sleepy or disengaged students
- 📊 **Analytics Dashboard**: Comprehensive engagement statistics and breakdowns
- 📈 **Historical Reports**: View past sessions with detailed student-level data
- 💾 **Database Storage**: PostgreSQL integration for persistent data storage
- ⚡ **High Performance**: Optimized for fast FPS while maintaining accuracy

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Webcam or video input device
- YOLOv11 trained model files (`.pt` or `.onnx` format)

## Installation

### 1. Clone the Repository

\`\`\`bash
git clone <your-repo-url>
cd engagetrack
\`\`\`

### 2. Create Virtual Environment

\`\`\`bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
\`\`\`

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Set Up PostgreSQL Database

Create a PostgreSQL database:

\`\`\`sql
CREATE DATABASE engagetrack;
\`\`\`

Update the database connection string in `app.py`:

\`\`\`python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/engagetrack'
\`\`\`

Replace `username` and `password` with your PostgreSQL credentials.

### 5. Add Your YOLOv11 Model

Place your trained YOLOv11 model file in the project root directory. The app will automatically look for:

- `engagement_detection_model.onnx` (ONNX format - recommended)
- `engagement_detection_model.pt` (PyTorch format)
- `best.onnx` (alternative ONNX name)

The model should be trained to detect engagement states such as:
- `engaged` / `focused`
- `neutral`
- `bored` / `distracted` / `looking_away`
- `sleepy`

### 6. Initialize Database

The database tables will be created automatically when you first run the application.

## Running the Application

### 1. Start the Flask Server

\`\`\`bash
python app.py
\`\`\`

The application will start on `http://127.0.0.1:5000`

### 2. Access the Application

Open your web browser and navigate to:

\`\`\`
http://localhost:5000
\`\`\`

### 3. Create an Account

1. Click "Sign up" on the login page
2. Enter your email and password
3. Sign in with your credentials

### 4. Start Detection

1. Navigate to the "Live" page
2. Click "Start Analysis" to begin detection
3. The system will:
   - Capture video from your webcam
   - Detect students and assign unique IDs
   - Display color-coded bounding boxes:
     - **Blue** (#60a5fa) for engaged students
     - **Gray** (#374151) for neutral students
     - **Red** (#c95d5e) for bored/sleepy students with "ALERT: SLEEPY!" text
   - Show real-time statistics
   - Trigger prominent red alerts for sleepy students
4. Click "Stop Analysis" when finished

### 5. View Reports

Navigate to the "Reports" page to:
- View past session summaries
- Analyze engagement breakdowns
- Review individual student performance

## Configuration

### Camera Settings

Adjust camera resolution and FPS in `app.py`:

\`\`\`python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
\`\`\`

### Detection Parameters

Modify confidence and IOU thresholds in the `generate_frames()` function:

\`\`\`python
results = model(frame, conf=0.4, iou=0.45, verbose=False, stream=False)
\`\`\`

- `conf`: Confidence threshold (0.0-1.0) - set to 0.4 as per YOLOv11 best practices
- `iou`: Intersection over Union threshold for NMS
- `verbose`: Set to False to suppress console output
- `stream`: Set to False for frame-by-frame processing

### Alert Threshold

Change the bored percentage that triggers alerts in `templates/live.html`:

\`\`\`javascript
if (data.bored > 30) {  // Change 30 to your desired threshold
    document.getElementById('sleepyAlert').classList.add('active');
}
\`\`\`

## YOLO Inference Pattern

The application uses the official Ultralytics YOLOv11 inference pattern:

\`\`\`python
from ultralytics import YOLO

# Load model with task specification
model = YOLO("engagement_detection_model.onnx", task='detect')

# Run inference
results = model(frame, conf=0.4, iou=0.45, verbose=False)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
\`\`\`

## Model Training

If you need to train your own YOLOv11 model:

1. Prepare a dataset with labeled images of students in different engagement states
2. Use the Ultralytics YOLOv11 training pipeline:

\`\`\`python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')

# Train the model
results = model.train(
    data='engagement_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Export the model to ONNX format
model.export(format='onnx')
\`\`\`

3. Place the exported model in the project root directory

## Performance Optimization

### For Faster FPS:

1. **Reduce video resolution**: Lower `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT`
2. **Use smaller model**: Use `yolo11n.pt` (nano) instead of larger variants
3. **Increase confidence threshold**: Higher `conf` value = fewer detections
4. **Reduce JPEG quality**: Lower the quality parameter in `cv2.imencode` (currently set to 80)
5. **Use ONNX format**: ONNX models are typically faster than PyTorch models

### For Better Accuracy:

1. **Increase video resolution**: Higher resolution captures more details
2. **Use larger model**: Use `yolo11m.pt` or `yolo11l.pt` for better accuracy
3. **Lower confidence threshold**: Lower `conf` value = more detections
4. **Fine-tune on your data**: Train the model on your specific classroom environment

## Troubleshooting

### Camera Not Working

- Check if your webcam is connected and not being used by another application
- Try changing the camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- On Linux, ensure you have proper permissions: `sudo usermod -a -G video $USER`

### Model Not Loading

- Ensure the model file is in the correct location (project root directory)
- Check that the model file name matches one of: `engagement_detection_model.onnx`, `engagement_detection_model.pt`, or `best.onnx`
- Verify the model was trained with compatible Ultralytics version (8.0+)
- Check console output for specific error messages

### Database Connection Error

- Verify PostgreSQL is running: `sudo service postgresql status` (Linux) or check Services (Windows)
- Check database credentials in `app.py`
- Ensure the database exists: `psql -l`
- Test connection: `psql -U username -d engagetrack`

### Slow Performance

- Reduce video resolution to 640x480 or lower
- Use a smaller YOLO model variant (yolo11n.pt)
- Close other applications using the camera or GPU
- Consider using GPU acceleration if available (CUDA-enabled GPU)
- Use ONNX format for faster inference

### No Detections Appearing

- Lower the confidence threshold (try `conf=0.3`)
- Ensure proper lighting in the room
- Check that the model was trained on similar data
- Verify the model classes match the expected engagement states

## Color Scheme

The application uses the following color palette:

- Background: `#111827` (Dark blue-gray)
- Primary Blue: `#60a5fa` (Engaged students)
- Secondary Blue: `#2563eb` (Buttons and accents)
- Gray: `#374151` (Neutral students)
- Alert Red: `#c95d5e` (Sleepy/bored students)
- White: `#ffffff` (Text and labels)

## Database Schema

### Users Table
- `id`: Primary key
- `email`: User email (unique)
- `password_hash`: Hashed password
- `created_at`: Account creation timestamp

### Sessions Table
- `id`: Primary key
- `user_id`: Foreign key to Users
- `start_time`: Session start timestamp
- `end_time`: Session end timestamp
- `duration`: Session duration in seconds
- `total_students`: Number of unique students detected
- `engaged_percentage`: Percentage of engaged detections
- `neutral_percentage`: Percentage of neutral detections
- `bored_percentage`: Percentage of bored/sleepy detections

### Detections Table
- `id`: Primary key
- `session_id`: Foreign key to Sessions
- `student_id`: Tracked student ID (consistent across frames)
- `timestamp`: Detection timestamp
- `engagement_state`: Engagement classification (engaged/neutral/bored/sleepy)
- `confidence`: Detection confidence score (0.0-1.0)

## Student ID Tracking

The application uses IoU (Intersection over Union) based tracking to assign consistent IDs to students:

- Each detected student receives a unique ID
- IDs persist across frames as long as the student remains visible
- Minimum IoU threshold of 0.3 for matching
- Trackers are cleaned up after 2 seconds of no detection
- Enables individual student analysis and historical tracking

## Security Notes

- **Change the SECRET_KEY**: Update `app.config['SECRET_KEY']` in `app.py` to a secure random string
- Use environment variables for sensitive configuration in production
- Implement HTTPS in production environments
- Add rate limiting for API endpoints
- Regularly update dependencies for security patches
- Use strong passwords for PostgreSQL database
- Consider implementing session timeouts

## Production Deployment

For production deployment:

1. Set `debug=False` in `app.run()`
2. Use a production WSGI server like Gunicorn:
   \`\`\`bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   \`\`\`
3. Set up a reverse proxy (Nginx/Apache)
4. Use environment variables for configuration
5. Enable HTTPS with SSL certificates
6. Set up proper logging and monitoring

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions, please open an issue on the GitHub repository.
