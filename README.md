# AI Depth Vision

This project uses the MiDaS depth estimation model to create real-time depth maps from a webcam feed. It generates two GIFs: one showing the original video feed and another showing the corresponding depth map.

![Depth Map GIF](depth_capture.gif)
![Normal Capture GIF](normal_capture.gif)

## Features

- Real-time depth estimation using MiDaS
- Visualization of depth map alongside original video feed
- Saves both depth map and original feed as GIFs

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Matplotlib
- Numpy
- Imageio

## Installation

1. Clone this repository: git clone https://github.com/yourusername/AiDepthVision.git
2. cd AiDepthVision
3. Install the required packages


- The program will open your webcam and start displaying the depth map alongside the original feed.
- Press 'q' while focused on the matplotlib window to stop the capture and save the GIFs.
- Two GIFs will be saved in your current directory: `depth_capture.gif` and `normal_capture.gif`.

## How it works

1. The script captures video from your webcam.
2. Each frame is processed through the MiDaS depth estimation model.
3. The resulting depth map is visualized using matplotlib.
4. Both the original frame and depth map are saved for GIF creation.
5. When you press 'q', the capture stops and the GIFs are generated.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/AiDepthVision/issues) if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- [MiDaS](https://github.com/isl-org/MiDaS) for the depth estimation model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [OpenCV](https://opencv.org/) for image processing
- [Matplotlib](https://matplotlib.org/) for visualization
