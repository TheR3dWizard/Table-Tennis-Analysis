import cv2
import time
import requests
import httpx
import os
import asyncio
from typing import AsyncIterator, Union, BinaryIO

# === CONFIG ===
video_path = "../Videos/game_1_30fps.mp4"
output_csv = "annotations.csv"
skip_seconds = 2
instructions = "SPACE=pause/resume | s=start | e=end | a/←=back | d/→=fwd | q=quit"
BASE_URL = "https://pv8zqdrn-6060.inc1.devtunnels.ms"
prompt = ""


# === INIT ===

async def main():

    upload_result = await upload_video_async(video_path)
    videoid = upload_result["filename"]  # use your returned video identifier
    print("Uploaded:", upload_result)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Couldn't open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    playback_speed = 1  # normal
    paused = False
    start_time = None
    annotations = []

    def frame_to_time(frame_idx):
        seconds = frame_idx / fps
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def draw_overlay(frame, frame_idx):
        time_str = frame_to_time(frame_idx)
        cv2.putText(frame, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        draw_overlay(frame, frame_idx)
        frame = cv2.resize(frame, (800, 600))  # Resize for better visibility
        cv2.imshow("Video Annotator", frame)

        key = cv2.waitKey(int(1000 / (fps * playback_speed))) & 0xFF

        # === CONTROLS ===
        if key == ord(' '):  # Pause/Resume
            paused = not paused

        elif key == ord('s'):
            start_time = frame_idx
            time_str = frame_to_time(start_time)
            print(f"[Start] at {time_str}")

        elif key == ord('e') and start_time:
            end_time = frame_idx
            end_time_str = frame_to_time(end_time)
            time_str = frame_to_time(start_time)

            prompt = prompt + f"From {time_str} to {end_time_str}, I have a question: "
            label = input(f"Prompt for event {time_str} to {end_time_str}: ")
            prompt = prompt + label + "\n"
            print("Asking question...")
            answer = await ask_question(
                prompt,
                videoid
            )
            print("Answer:", answer)

            start_time = None



        elif key in [ord('a'), 81]:  # Left arrow / a
            back_frame = max(0, frame_idx - int(skip_seconds * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, back_frame)

        elif key in [ord('d'), 83]:  # Right arrow / d
            fwd_frame = min(total_frames - 1, frame_idx + int(skip_seconds * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fwd_frame)

        elif key == ord('+'):
            playback_speed = min(4.0, playback_speed + 0.25)
            print(f"Playback speed: {playback_speed:.2f}x")

        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed - 0.25)
            print(f"Playback speed: {playback_speed:.2f}x")

        elif key in [ord('q'), 27]:  # Quit on q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


CHUNK_SIZE = 1024 * 1024  # 1 MB


async def file_chunker(file_obj: BinaryIO) -> AsyncIterator[bytes]:
    """Yield the file in chunks without reading it all at once."""
    while True:
        chunk = file_obj.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk

async def upload_video_async(file_path: str) -> dict:
    async with httpx.AsyncClient(timeout=None) as client:
        with open(file_path, "rb") as f:
            async def gen():
                async for chunk in file_chunker(f):
                    yield chunk

            headers = {"Content-Type": "application/octet-stream"}

            response = await client.post(
                f"{BASE_URL}/upload-video",
                headers=headers,
                content=gen()  # pass generator directly; DO NOT call .read()
            )

        response.raise_for_status()
        return response.json()



async def ask_question(question, videoid, start_frame=None, end_frame=None):
    """
    Send a question to /ask-question asynchronously.
    You may optionally embed frame range in the question text.
    """

    url = f"{BASE_URL}/ask-question"

    # If frame data is included, append to question string (your backend extracts)
    if start_frame is not None and end_frame is not None:
        question = f"{question} (from {start_frame} to {end_frame})"

    payload = {
        "question": question,
        "videoid": videoid
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    asyncio.run(main())
