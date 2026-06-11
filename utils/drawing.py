import cv2
from mediapipe.tasks.python.vision import PoseLandmarksConnections

# رنگ‌های skeleton
COLOR_LANDMARK = (0, 255, 0)      # سبز برای نقاط مفاصل
COLOR_CONNECTION = (255, 255, 0)  # زرد برای خطوط بین مفاصل


def draw_skeleton(frame, results):
    """
    رسم skeleton روی تصویر

    در نسخه جدید mediapipe نمیشه از mp_drawing قدیمی استفاده کرد
    پس دستی رسم میکنیم:
        1. مختصات نرمال (0 تا 1) رو به pixel تبدیل میکنیم
        2. دایره روی هر نقطه مفصل رسم میکنیم
        3. خط بین نقاط متصل رسم میکنیم

    connections از PoseLandmarksConnections.POSE_LANDMARKS میاد
    که لیستی از جفت‌های (start, end) هست
    """
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return frame

    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks[0]

    # تبدیل مختصات نرمال به pixel و رسم نقاط
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1)

    # رسم خطوط بین مفاصل
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        s = conn.start
        e = conn.end
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], COLOR_CONNECTION, 2)

    return frame


def draw_text(frame, text, position=(30, 50), font_scale=1.0, color=(0, 255, 0), thickness=2):
    """
    نوشتن متن روی تصویر
    color به فرمت BGR هست، مثلاً (0,255,0) یعنی سبز
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    return frame


def draw_status(frame, is_detected):
    """
    نمایش وضعیت تشخیص بدن در گوشه تصویر
    سبز = بدن پیدا شده | قرمز = بدن پیدا نشده
    """
    if is_detected:
        draw_text(frame, "Pose: Detected", position=(30, 50), color=(0, 255, 0))
    else:
        draw_text(frame, "Pose: Not Detected", position=(30, 50), color=(0, 0, 255))
    return frame