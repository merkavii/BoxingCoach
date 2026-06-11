import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
import numpy as np
import urllib.request
import os

# مسیر ذخیره فایل مدل
MODEL_PATH = "models/pose_landmarker.task"


def download_model():
    """
    دانلود فایل مدل pose از سرور mediapipe
    این فایل فقط یه بار دانلود میشه و ذخیره میمونه
    """
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("در حال دانلود مدل pose_landmarker...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("مدل با موفقیت دانلود شد.")
    else:
        print("مدل از قبل موجوده.")


def create_pose_model():
    """
    ساخت و برگردوندن مدل PoseLandmarker

    در نسخه جدید mediapipe باید:
        1. فایل مدل .task رو دانلود کنی
        2. از PoseLandmarkerOptions برای تنظیمات استفاده کنی
        3. مدل رو با create_from_options بسازی

    running_mode=VIDEO یعنی برای ویدیو زنده بهینه‌سازی شده
    """
    download_model()

    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,  # برای پردازش فریم به فریم
        num_poses=1,                             # فقط یه نفر رو تشخیص بده
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose = PoseLandmarker.create_from_options(options)
    print("مدل pose آماده‌ست.")
    return pose


def process_frame(pose, frame_rgb, timestamp_ms):
    """
    پردازش یک فریم توسط mediapipe

    در نسخه جدید:
        - باید timestamp بدی (زمان فریم به میلی‌ثانیه)
        - ورودی باید mp.Image باشه نه numpy array مستقیم
        - از detect_for_video استفاده میکنیم چون running_mode=VIDEO هست

    timestamp_ms → زمان فریم فعلی به میلی‌ثانیه
    """
    # تبدیل numpy array به فرمت mp.Image که mediapipe میفهمه
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # پردازش با timestamp
    results = pose.detect_for_video(mp_image, timestamp_ms)
    return results


def extract_keypoints(results):
    """
    استخراج مختصات 33 نقطه بدن از نتایج mediapipe

    در نسخه جدید results.pose_landmarks یه لیسته (چون میتونه چند نفر باشه)
    ما فقط نفر اول رو میگیریم: results.pose_landmarks[0]

    خروجی: آرایه numpy با 99 عدد (33 نقطه × x, y, z)
    """
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        # گرفتن نقاط نفر اول
        landmarks = results.pose_landmarks[0]
        keypoints = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ]).flatten()
    else:
        # اگه بدنی پیدا نشد آرایه صفر برمیگردونیم
        keypoints = np.zeros(33 * 3)

    return keypoints