import cv2
import streamlit as st
import streamlit_webrtc
import torch
import av
import threading
import time

st.title("(Orange) Safety Helmet Detector")

# Load the YOLOv5 model
# Replace 'yolov5s.pt' with your trained/custom model path if applicable
model_path = 'last.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

lock = threading.Lock()

class VideoProcessor(streamlit_webrtc.VideoTransformerBase):
    def __init__(self) -> None:
        self.helmet_amount = 0
        # Assuming self.model is properly initialized here

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.reformat(frame.width / 4, frame.height / 4).to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = model(img_rgb)

        # Parse detection results and plot them directly on the frame
        detections = results.pandas().xyxy[0]  # Get the detections as a Pandas DataFrame
        self.helmet_amount = 0
        
        helmet = 0
        for _, row in detections.iterrows():
            # Extract bounding box coordinates and other info
            x1, y1, x2, y2, conf, cls, name = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
                row["confidence"],
                int(row["class"]),
                row["name"],
            )
            if(0.5<conf):
                # Increment helmet
                helmet += 1

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Put label and confidence score
                # label = f"{name} {conf:.2f}"
                # cv2.putText(
                #     img,
                #     label,
                #     (x1, y1 - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 255, 0),
                #     2,
                # )
        with lock:
            self.helmet_amount = helmet

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    
camera_stream = streamlit_webrtc.webrtc_streamer(
    key="webcam",
    video_processor_factory=VideoProcessor,
    sendback_audio=False,
    async_processing=True,
    )

text_display = st.empty()

while camera_stream.state.playing:
    time.sleep(0.1)
    with lock:
        text_display.write("Amount of helmet: " \
                           + str(camera_stream.video_processor.helmet_amount))
