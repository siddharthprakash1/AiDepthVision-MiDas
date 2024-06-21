import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Load the MiDaS model
model_type = "DPT_Large"  
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

# here this code is checking for the gpu 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

#here the if else is unnecsary but I have added so that I can check the performance of each dpt 

cap = cv2.VideoCapture(0)
#here i did not go with display of opencv as it doesnot  have support of gpu 


plt.ion()  # Turn on interactive mode for matplotlib

# Initialize lists to store frames for GIFs
depth_frames = []
normal_frames = []

# Create a figure and axes for plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Flag to control the main loop
running = True

def on_key(event):
    global running
    if event.key == 'q':
        running = False

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)

while running and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Make prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Convert depth map to RGB for GIF saving
    depth_map_rgb = plt.cm.inferno(depth_map)
    depth_map_rgb = (depth_map_rgb[:, :, :3] * 255).astype(np.uint8)

    # Append frames to lists for GIF creation
    depth_frames.append(depth_map_rgb)
    normal_frames.append(img)

    # Display depth map and original frame using matplotlib
    ax1.clear()
    ax1.imshow(depth_map, cmap='inferno')
    ax1.set_title('Depth Map')

    ax2.clear()
    ax2.imshow(img)
    ax2.set_title('Original Frame')

    plt.draw()
    plt.pause(0.001)  # Briefly pause to update the plots

cap.release()
plt.ioff()  # Turn off interactive mode
plt.close()  # Close the matplotlib window

# Save depth frames as GIF
imageio.mimsave('depth_capture.gif', depth_frames, fps=30)

# Save normal frames as GIF
imageio.mimsave('normal_capture.gif', normal_frames, fps=30)

print("GIFs saved successfully!")