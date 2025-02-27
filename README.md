# Gesture Control System

> Made with ❤️ by Shaurya Nandecha

A sophisticated Python-based gesture control system that allows you to control your computer's mouse cursor using hand gestures captured through your webcam. This project combines computer vision and machine learning to create a natural and intuitive way to interact with your computer.

## Features

- Control mouse cursor with index finger movements
- Left click and drag with index finger + thumb pinch
- Right click with middle finger + thumb pinch
- Double click with quick double pinch
- Activate/deactivate gesture control with "Yo" sign (index and pinky up)
- Smooth cursor movement with enhanced gesture detection
- Debug visualization mode

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`):
  - OpenCV
  - MediaPipe
  - PyAutoGUI
  - NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Shauryan28/Gesture-Control-Cursor.git
cd Gesture-Control-Cursor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python gesture_control.py
```

2. To activate/deactivate gesture control:
   - Make a "Yo" sign (index finger and pinky up, other fingers down)
   - Hold it for 1 second
   - Status will change from "Inactive" (red) to "Active" (green)

3. When active:
   - Move index finger to control cursor
   - Pinch index finger and thumb for left click
   - Hold pinch and move for drag operations
   - Quick double pinch for double click
   - Pinch middle finger and thumb for right click

4. To quit:
   - Press 'q' while the webcam window is focused
   - Or close the webcam window

## Debug Visualization

The program includes visual feedback to help with gesture recognition:
- Green circles: Extended fingers
- Red circles: Folded fingers
- Blue circles: Thumb position when near middle/ring fingers
- Purple line: Appears when "Yo" sign is detected

## Citation and Credits

If you use this project in your work, please provide appropriate credit by:

1. Adding a citation in your README:
```markdown
This project uses [Gesture Control System](https://github.com/Shauryan28/Gesture-Control-Cursor) by [Shaurya Nandecha](https://github.com/Shauryan28).
```

2. Adding a star ⭐ to the [original repository](https://github.com/Shauryan28/Gesture-Control-Cursor)

3. If you create a derivative work, please maintain the original credit in your code and documentation:
```python
"""
Based on Gesture Control System
Original work by Shaurya Nandecha (https://github.com/Shauryan28)
"""
```

For academic use, please cite as:
```bibtex
@software{nandecha2024gesture,
    author = {Shaurya Nandecha},
    title = {Gesture Control System},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/Shauryan28/Gesture-Control-Cursor}
}
```

## Author

**Shaurya Nandecha**
- GitHub: [@Shauryan28](https://github.com/Shauryan28)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
If you found this project helpful or interesting, please consider giving it a ⭐️ on GitHub! 