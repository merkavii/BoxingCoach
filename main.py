import cv2
import time

from utils.camera import connect_camera, read_frame, release_camera
from utils.drawing import draw_skeleton, draw_status
from core.pose_estimator import create_pose_model, process_frame, extract_keypoints
from config import CAMERA_INDEX


def main():
    # اتصال به دوربین
    cap = connect_camera(index=CAMERA_INDEX)

    # ساخت مدل pose (اگه اول باره، مدل دانلود میشه)
    pose = create_pose_model()

    print("برنامه شروع شد. برای خروج کلید 'q' رو بزن.")

    while cap.isOpened():
        ret, frame = read_frame(cap)
        if not ret:
            print("خطا در خواندن فریم.")
            break

        # گرفتن timestamp فعلی به میلی‌ثانیه
        # mediapipe جدید برای detect_for_video به timestamp نیاز داره
        timestamp_ms = int(time.time() * 1000)

        # تبدیل BGR به RGB برای mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # پردازش فریم
        results = process_frame(pose, frame_rgb, timestamp_ms)

        # استخراج keypoints
        keypoints = extract_keypoints(results)

        # رسم روی تصویر
        frame = draw_skeleton(frame, results)
        is_detected = results.pose_landmarks is not None and len(results.pose_landmarks) > 0
        frame = draw_status(frame, is_detected)

        if is_detected:
            print(f"keypoints: {keypoints[:6].round(3)} ...")

        cv2.imshow("Boxing Coach - Phase 1", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("برنامه بسته شد.")
            break

    release_camera(cap)


if __name__ == "__main__":
    main()