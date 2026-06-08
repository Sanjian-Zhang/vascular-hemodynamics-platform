from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
except ImportError:
    print("Missing dependency: selenium")
    print("Install with: pip install selenium")
    sys.exit(1)


HTML_FILE = Path(__file__).with_name("lc_3_interactive_velocity_viewer.html")


PHASE_DEFINITIONS = [
    {"key": "top-overview", "name": "Top Overview", "start": 0.0, "end": 0.14},
    {"key": "perimeter-orbit", "name": "Perimeter Orbit", "start": 0.14, "end": 0.30},
    {"key": "core-approach", "name": "Core Approach", "start": 0.30, "end": 0.44},
    {"key": "interior-grove", "name": "Interior Grove", "start": 0.44, "end": 0.88},
    {"key": "exit-reveal", "name": "Exit Reveal", "start": 0.88, "end": 1.0},
]


def build_driver(width: int, height: int):
    browser_errors = []

    chrome_options = ChromeOptions()
    for arg in (
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
        f"--window-size={width},{height}",
    ):
        chrome_options.add_argument(arg)

    try:
        return webdriver.Chrome(options=chrome_options)
    except Exception as exc:
        browser_errors.append(f"Chrome unavailable: {exc}")

    edge_options = EdgeOptions()
    for arg in (
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
        f"--window-size={width},{height}",
    ):
        edge_options.add_argument(arg)

    try:
        return webdriver.Edge(options=edge_options)
    except Exception as exc:
        browser_errors.append(f"Edge unavailable: {exc}")

    details = "\n".join(browser_errors) if browser_errors else "No compatible browser found."
    raise RuntimeError(
        "Unable to start a headless browser for frame export.\n"
        "Install Google Chrome or Microsoft Edge, then rerun.\n"
        f"{details}"
    )


def wait_for_export_api(driver, timeout_seconds: float):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            ready = driver.execute_script(
                "return Boolean(window.viewerExport && window.viewerExport.ready);"
            )
            if ready:
                return
        except WebDriverException as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"viewerExport API was not ready in time. Last error: {last_error}")


def decode_data_url(data_url: str) -> bytes:
    payload = data_url.split(",", 1)[1]
    return base64.b64decode(payload)


def export_frames(
    output_dir: Path,
    frame_count: int,
    duration_seconds: float,
    width: int,
    height: int,
    wait_seconds: float,
    start_index: int,
    seed: int,
    phase_key: str | None = None,
):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"HTML viewer not found: {HTML_FILE}")

    if phase_key:
        phase = next((p for p in PHASE_DEFINITIONS if p["key"] == phase_key), None)
        if not phase:
            raise ValueError(f"Unknown phase: {phase_key}")
        print(f"Exporting phase: {phase['name']} ({phase['start']:.2f} to {phase['end']:.2f})")
        progress_start = phase["start"]
        progress_end = phase["end"]
    else:
        progress_start = 0.0
        progress_end = 1.0
        print("Exporting full animation")

    url = HTML_FILE.resolve().as_uri()
    driver = build_driver(width=width, height=height)
    try:
        driver.get(url)
        wait_for_export_api(driver, timeout_seconds=wait_seconds)

        driver.execute_script(
            """
            window.viewerExport.reset({
              seed: arguments[0],
              startProgress: arguments[1],
            });
            """,
            seed,
            progress_start,
        )

        digits = max(4, len(str(start_index + frame_count - 1)))
        for frame_offset in range(frame_count):
            local_progress = frame_offset / max(frame_count - 1, 1)
            global_progress = progress_start + (progress_end - progress_start) * local_progress
            
            data_url = driver.execute_script(
                """
                return window.viewerExport.renderFrame(
                  arguments[0],
                  arguments[1],
                  {
                    durationSeconds: arguments[2],
                    seed: arguments[3],
                    progress: arguments[4]
                  }
                );
                """,
                frame_offset,
                frame_count,
                duration_seconds,
                seed,
                global_progress,
            )
            frame_number = start_index + frame_offset
            frame_path = output_dir / f"{frame_number:0{digits}d}.png"
            frame_path.write_bytes(decode_data_url(data_url))

            if frame_offset == 0 or (frame_offset + 1) % 20 == 0 or frame_offset == frame_count - 1:
                print(f"Saved frame {frame_number} -> {frame_path}")

        print(f"Export complete: {frame_count} frames written to {output_dir.resolve()}")
    finally:
        driver.quit()


def detect_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    for path in [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"D:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(path).exists():
            return path
    return None


def frames_to_video(
    frames_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    crf: int = 18,
    preset: str = "medium",
):
    ffmpeg_path = detect_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and add it to PATH, "
            "or specify the full path in detect_ffmpeg()."
        )

    frame_pattern = str(frames_dir / "%04d.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"Running ffmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    
    print(f"Video saved to {output_path.resolve()}")


def list_phases():
    print("Available phases:")
    for phase in PHASE_DEFINITIONS:
        print(f"  {phase['key']:20} {phase['name']} (progress: {phase['start']:.2f} - {phase['end']:.2f})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the vessel viewer animation as numbered PNG frames or video."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="frames",
        help="Output directory for PNG frames (default: 'frames').",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=180,
        help="Number of frames to export (default: 180).",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=6.0,
        help="Virtual duration in seconds spread across the exported frames (default: 6.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Export width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Export height in pixels (default: 1080).",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=15.0,
        help="Maximum seconds to wait for the viewer to load (default: 15.0).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting frame number for output naming (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deterministic particle playback (default: 12345).",
    )
    parser.add_argument(
        "--phase",
        "-p",
        choices=[p["key"] for p in PHASE_DEFINITIONS],
        help="Export only a specific phase instead of the full animation.",
    )
    parser.add_argument(
        "--list-phases",
        "-l",
        action="store_true",
        help="List all available phases and exit.",
    )
    parser.add_argument(
        "--video",
        "-v",
        action="store_true",
        help="Export directly to video (requires ffmpeg).",
    )
    parser.add_argument(
        "--video-output",
        default="output.mp4",
        help="Output video file path (default: 'output.mp4').",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for video output (default: 30).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF quality for video (lower = better quality, 0-51, default: 18).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_phases:
        list_phases()
        return

    output_dir = Path(args.output)
    
    export_frames(
        output_dir=output_dir,
        frame_count=args.frames,
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        wait_seconds=args.wait,
        start_index=args.start_index,
        seed=args.seed,
        phase_key=args.phase,
    )

    if args.video:
        video_output = Path(args.video_output)
        frames_to_video(
            frames_dir=output_dir,
            output_path=video_output,
            fps=args.fps,
            crf=args.crf,
        )


if __name__ == "__main__":
    main()