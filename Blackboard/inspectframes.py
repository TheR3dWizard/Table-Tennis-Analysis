import cv2
import sys


def play_video_with_frame_counter(video_path, start_frame=0, end_frame=None, pause_frames=None):
    """
    Play a video and display the current frame number on the top right corner.

    Args:
        video_path: Path to the video file
        start_frame: Frame number to start playback from (default: 0)
        end_frame: Frame number to end playback at (default: None, plays until end)
        pause_frames: Optional list of frame IDs to pause at. When current frame is in this list,
                      playback will pause and wait for space key before continuing (default: None)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate and set end_frame
    if end_frame is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(end_frame, total_frames - 1)

    # Validate start_frame
    start_frame = max(0, min(start_frame, total_frames - 1))

    # Convert pause_frames to set for faster lookup
    pause_frames_set = set(pause_frames) if pause_frames else set()

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Playing frames: {start_frame} to {end_frame}")
    if pause_frames_set:
        print(f"Will pause at frames: {sorted(pause_frames_set)}")
    print("Press 'q' to quit, 'p' to pause/resume, 'r' to restart, SPACE to continue from pause frame")

    frame_number = start_frame
    paused = False
    paused_at_frame = False  # Track if we're paused at a specific frame

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("End of video reached")
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Check if we've reached the end_frame
            if frame_number > end_frame:
                print(f"Reached end frame: {end_frame}")
                break

            # Check if current frame is in pause_frames_set
            if frame_number in pause_frames_set:
                paused = True
                paused_at_frame = True
                print(f"Paused at frame {frame_number} (pause frame). Press SPACE to continue.")

        # Create a copy to draw on
        display_frame = frame.copy()

        # Get frame dimensions
        height, width = display_frame.shape[:2]

        # Prepare text
        text = f"Frame: {frame_number}/{end_frame}"
        if paused_at_frame:
            text += " [BOUNCE DETECTED]"

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (0, 255, 0)  # Green
        bg_color = (0, 0, 0)  # Black background

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Position on top right with padding
        padding = 10
        x = width - text_width - padding
        y = text_height + padding

        # Draw background rectangle
        cv2.rectangle(
            display_frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1,
        )

        # Draw text
        cv2.putText(
            display_frame,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        # If paused at a specific frame, add a visual indicator
        if paused_at_frame:
            pause_text = "Press SPACE to continue"
            pause_font_scale = 0.7
            (pause_text_width, pause_text_height), pause_baseline = cv2.getTextSize(
                pause_text, font, pause_font_scale, 1
            )
            pause_x = (width - pause_text_width) // 2
            pause_y = height - 30
            
            # Draw background for pause text
            cv2.rectangle(
                display_frame,
                (pause_x - 5, pause_y - pause_text_height - 5),
                (pause_x + pause_text_width + 5, pause_y + pause_baseline + 5),
                (0, 0, 255),  # Red background
                -1,
            )
            # Draw pause text
            cv2.putText(
                display_frame,
                pause_text,
                (pause_x, pause_y),
                font,
                pause_font_scale,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA,
            )

        # Display the frame
        cv2.imshow("Video Player", display_frame)

        # Wait for key press
        # If paused at a specific frame, wait indefinitely for space key
        # Otherwise, use normal timing or wait for manual pause
        if paused_at_frame:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(1000 / fps) if not paused else 0) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord(" ") and paused_at_frame:
            # Space key pressed when paused at a specific frame
            paused = False
            paused_at_frame = False
            print(f"Continuing from frame {frame_number}...")
        elif key == ord("p"):
            # Only allow manual pause/resume if not paused at a specific frame
            if not paused_at_frame:
                paused = not paused
                print("Paused" if paused else "Resumed")
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame
            paused = False
            paused_at_frame = False
            print(f"Restarting from frame {start_frame}...")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def rewritefps():
    cap = cv2.VideoCapture(
        "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/testscripts/game_1.mp4"
    )
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/testscripts/game_1_60.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        240,
        (frame_width, frame_height),
    )
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        print(i)
        if i % 4 == 0:
            writer.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/assets/short-slow-rally.mp4"
    play_video_with_frame_counter(video_path, pause_frames=[32, 54, 89, 111, 147])
