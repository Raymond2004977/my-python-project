import streamlit as st
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO


st.set_page_config(
    page_title="Border Area Monitoring System",
    layout="wide"
)


st.title("Border Area Monitoring System Using AI-Powered Drone")
st.write("Real-Time Human Detection and Motion Tracking using YOLOv8")


st.sidebar.header("System Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4,
    0.05
)


uploaded_video = st.file_uploader(
    "Upload Drone Surveillance Video",
    type=["mp4", "avi", "mov", "mkv"]
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


def convert_to_playable_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run([
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ], check=True)


if uploaded_video is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    st.subheader("Uploaded Video")
    st.video(temp_input.name)

    if st.button("Start Human Detection"):
        st.info("Processing video, please wait...")

        cap = cv2.VideoCapture(temp_input.name)

        if not cap.isOpened():
            st.error("Could not open video file.")
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0:
                fps = 25

            raw_output_path = "border_monitoring_ui_output_raw.mp4"
            playable_output_path = "border_monitoring_ui_output_playable.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (frame_width, frame_height)
            )

            if not out.isOpened():
                cap.release()
                st.error("Could not create output video file.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_number = 0
                total_human_detections = 0
                alert_detected = False
                max_humans_in_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1
                    human_count = 0

                    results = model.track(
                        frame,
                        persist=True,
                        classes=[0],
                        conf=confidence_threshold,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            human_count += 1
                            total_human_detections += 1
                            alert_detected = True

                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0])

                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id[0])

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                            label = f"Human {confidence:.2f}"

                            if track_id is not None:
                                label = f"ID {track_id} | Human {confidence:.2f}"

                            cv2.putText(
                                frame,
                                label,
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                    max_humans_in_frame = max(max_humans_in_frame, human_count)

                    cv2.putText(
                        frame,
                        f"Human Count: {human_count}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2
                    )

                    if human_count > 0:
                        cv2.putText(
                            frame,
                            "ALERT: Human Detected!",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                    out.write(frame)

                    if total_frames > 0:
                        progress = int((frame_number / total_frames) * 100)
                        progress_bar.progress(min(progress, 100))

                    status_text.write(
                        f"Processing frame {frame_number} of {total_frames}"
                    )

                cap.release()
                out.release()

                status_text.write("Converting output video to playable format...")

                try:
                    convert_to_playable_mp4(
                        raw_output_path,
                        playable_output_path
                    )

                    st.success("Video processing completed successfully.")

                    st.subheader("Detection Result Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Total Frames", total_frames)
                    col2.metric("Total Human Detections", total_human_detections)
                    col3.metric("Maximum Humans in Frame", max_humans_in_frame)
                    col4.metric(
                        "Alert Status",
                        "Human Detected" if alert_detected else "No Human Detected"
                    )

                    st.subheader("Processed Output Video")
                    st.video(playable_output_path)

                    st.write("Output saved as:", playable_output_path)

                except Exception as error:
                    st.error("Video was processed, but conversion failed.")
                    st.write("Error:", error)
                    st.write("Raw output saved as:", raw_output_path)

else:
    st.warning("Please upload a drone surveillance video to begin.")
