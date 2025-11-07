import cv2
import time
import requests
import httpx
import os
import asyncio
from typing import AsyncIterator, Union, BinaryIO
from pprint import pprint
# === CONFIG ===
video_path = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/assets/short-slow-rally.mp4"
output_csv = "annotations.csv"
skip_seconds = 2
instructions = "SPACE=pause/resume | s=start | e=end | a/←=back | d/→=fwd | q=quit"
BASE_URL = "http://localhost:6060"
prompt = ""

# === INIT ===

async def main():

    upload_result = await upload_video_async(video_path)
    videoid = 3
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

            prompt_text = f"From frame {start_time} to frame {end_time}, "
            label = input(f"Question for frames {start_time} to {end_time}: ")
            prompt_text = label
            print("Asking question...")
            answer = await ask_question(
                prompt_text,
                videoid
            )
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            
            # Extract LLM response
            llm_response = None
            if 'analysis' in answer and 'llmans' in answer['analysis']:
                llmans = answer['analysis']['llmans']
                if llmans and len(llmans) > 0:
                    llm_data = llmans[0]
                    # Try to extract response - handle both dict and string representations
                    if isinstance(llm_data, dict):
                        llm_response = llm_data.get('response', None)
                    else:
                        # Extract from string representation
                        llm_str = str(llm_data)
                        # Find the response="..." part (handle escaped quotes)
                        if 'response="' in llm_str:
                            start_idx = llm_str.find('response="') + len('response="')
                            # Find the closing quote, handling escaped quotes
                            end_idx = start_idx
                            while end_idx < len(llm_str):
                                if llm_str[end_idx] == '"' and (end_idx == start_idx or llm_str[end_idx-1] != '\\'):
                                    break
                                end_idx += 1
                            if end_idx > start_idx:
                                llm_response = llm_str[start_idx:end_idx]
                                # Unescape common escape sequences
                                llm_response = llm_response.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            
            # Extract ball bounces
            ball_bounces = answer.get('ball_bounces', [])
            
            # Print LLM response
            if llm_response:
                print("\n📝 LLM Analysis:")
                print("-" * 80)
                print(llm_response)
            else:
                print("\n⚠️  No LLM response found")
            
            # Print bounces
            if ball_bounces:
                print("\n🏓 Ball Bounces:")
                print("-" * 80)
                print(f"Total bounces: {len(ball_bounces)}")
                print(f"Bounce frames: {ball_bounces}")
                
                # If detailed bounce info is available, show it
                if 'analysis' in answer and 'bounces' in answer['analysis']:
                    bounces_detail = answer['analysis']['bounces']
                    print("\nDetailed bounce information:")
                    for bounce_id, bounce_info in bounces_detail.items():
                        frame = bounce_info.get('bounceFrame', 'N/A')
                        segment = bounce_info.get('segment', 'N/A')
                        print(f"  Bounce {bounce_id}: Frame {frame} - Segment: {segment}")
            else:
                print("\n⚠️  No ball bounces found")
            
            print("="*80 + "\n")

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


async def upload_video_async(file_path: str, filename: str = None):
    """Upload video file using streaming to avoid memory issues."""
    filename = filename or os.path.basename(file_path)
    
    url = f"{BASE_URL}/upload-video"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Open file and upload with streaming
        with open(file_path, 'rb') as f:
            files = {
                'file': (filename, f, 'video/mp4')
            }
            data = {
                'filename': filename
            }
            
            response = await client.post(url, files=files, data=data)
    
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)

    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    asyncio.run(main())