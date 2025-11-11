# Detection Threshold Settings

## Current Configuration (Optimized to Reduce False Positives)

### YOLO Direct Detection
- **Confidence Threshold**: 0.4 (40%)
- **IOU Threshold**: 0.5
- Only shows detections with 40%+ confidence

### OpenCV Face Detection (Fallback)
- **Scale Factor**: 1.1
- **Min Neighbors**: 7 (increased from 5 for fewer false positives)
- **Min Face Size**: 100x100 pixels (increased from 80x80)
- **Max Faces**: 10 per frame

### YOLO on Face Crops
- **Confidence Threshold**: 0.3 (30%)
- Filters out any detection below 30% confidence
- Only processes the best detection per face

## What This Means

✅ **Fewer false positives** - Only high-confidence detections shown
✅ **Better accuracy** - Stricter face detection parameters
✅ **Performance** - Limited to 10 faces max per frame
✅ **Cleaner display** - No overlapping boxes on same face

## If You Need to Adjust

### Too Few Detections?
Lower the thresholds in `app.py`:
- Line ~305: Change `conf=0.4` to `conf=0.3`
- Line ~327: Change `minNeighbors=7` to `minNeighbors=5`
- Line ~345: Change `conf=0.3` to `conf=0.2`

### Too Many Detections?
Increase the thresholds:
- Line ~305: Change `conf=0.4` to `conf=0.5`
- Line ~327: Change `minNeighbors=7` to `minNeighbors=9`
- Line ~345: Change `conf=0.3` to `conf=0.4`

### Adjust Max Faces
Line ~330: Change `faces = faces[:10]` to your desired limit

## Testing Tips

1. **Good lighting** helps detection accuracy
2. **Face the camera** directly for best results
3. **Distance**: 2-6 feet from camera is optimal
4. **Multiple people**: Works best with 1-5 people in frame
