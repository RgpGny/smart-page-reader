# Smart Page Reader - SVM Decision Boundary Visualization

A machine learning application that uses Support Vector Machine (SVM) to create decision boundaries between different shapes (circles and crosses) marked on camera-captured images.

## 🚀 Features

- **Real-time Image Capture**: Capture images directly from your camera
- **Interactive Shape Marking**: Mark circles and crosses on captured images with an intuitive interface
- **SVM Classification**: Train a Support Vector Machine model to classify between two shape types
- **Decision Boundary Visualization**: Visualize the SVM decision boundary on the original image
- **Data Export**: Save marked coordinates to CSV format for analysis
- **Test Image Generation**: Create synthetic test images for development

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Project2
```

2. Install required dependencies:
```bash
pip install opencv-python numpy matplotlib scikit-learn pandas
```

## 📖 Usage

### Running the Main Application

```bash
python smart_page_reader.py
```

The application follows these steps:
1. **Camera Capture**: Press SPACE to capture an image, ESC to exit
2. **Shape Marking**: 
   - Press 'c' to switch to circle marking mode
   - Press 'x' to switch to cross marking mode
   - Left-click to mark shapes on the image
   - Press SPACE to finish marking and save data
3. **Model Training**: The SVM model trains automatically on marked data
4. **Visualization**: View the decision boundary overlaid on your image

### Creating Test Images

```bash
python create_test_image.py
```

This generates a test image (`input_image.jpg`) with predefined circles and crosses for testing purposes.

## 📁 Project Structure

```
Project2/
├── smart_page_reader.py     # Main application with SmartPageReader class
├── create_test_image.py     # Test image generator
├── dataset.csv              # Saved coordinates and shape labels
├── decision_boundary_*.png  # Generated decision boundary visualizations
└── README.md               # Project documentation
```

## 🔧 Core Components

### SmartPageReader Class

The main class that handles all functionality:

- **`capture_image()`**: Captures images from the camera
- **`mark_shapes()`**: Interactive shape marking interface
- **`save_coordinates()`**: Exports marked data to CSV
- **`train_model()`**: Trains SVM model on marked coordinates
- **`draw_decision_boundary()`**: Visualizes decision boundary

### Key Features

- **SVM Algorithm**: Uses RBF (Radial Basis Function) kernel for non-linear classification
- **Data Preprocessing**: Automatic feature scaling using StandardScaler
- **Interactive UI**: Real-time shape marking with visual feedback
- **Export Functionality**: Saves results as timestamped PNG files

## 📊 Data Format

The application saves marked coordinates in CSV format:

```csv
x,y,shape
477,394,circle
348,272,cross
...
```

- **x, y**: Pixel coordinates of marked points
- **shape**: Either "circle" or "cross"

## 🎮 Controls

### Camera Mode
- **SPACE**: Capture photo
- **ESC**: Exit without capturing

### Shape Marking Mode
- **Left Click**: Mark shape at cursor position
- **'c' Key**: Switch to circle marking mode
- **'x' Key**: Switch to cross marking mode
- **SPACE**: Finish marking and save data
- **ESC**: Cancel marking (clears all marks)

## 🔬 Technical Details

### Machine Learning
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Features**: 2D coordinates (x, y)
- **Classes**: Binary classification (circle vs cross)

### Image Processing
- **Library**: OpenCV
- **Input**: Camera capture or static images
- **Output**: Annotated images with decision boundaries

### Visualization
- **Decision Boundary**: Blue contour line showing SVM classification boundary
- **Circles**: Green circles with center points
- **Crosses**: Red cross markers with center points

## 📈 Example Output

The application generates images showing:
- Original marked shapes (green circles, red crosses)
- SVM decision boundary (blue line)
- Clear separation between different shape regions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**: Ensure your camera is connected and not being used by another application
2. **OpenCV import error**: Install OpenCV using `pip install opencv-python`
3. **No shapes marked**: Make sure to mark at least one circle and one cross before training
4. **Decision boundary not visible**: Ensure sufficient contrast between shape regions

### Requirements Check

Make sure all dependencies are installed:
```bash
pip list | grep -E "(opencv|numpy|matplotlib|scikit|pandas)"
```

## 🎯 Future Enhancements

- Support for additional shape types
- Real-time classification mode
- Model persistence and loading
- Batch processing capabilities
- Advanced visualization options

---

*This project demonstrates practical application of machine learning concepts in computer vision and interactive data labeling.* 