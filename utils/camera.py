import cv2


def connect_camera(index=0):
    """
    اتصال به دوربین و برگردوندن شیء cap

    index=0 معمولاً دوربین پیش‌فرض لپ‌تاپه
    index=1 یا 2 معمولاً DroidCam هست (باید امتحان کنی)
    اگه با IP متصل میشی، به جای عدد آدرس بده:
        connect_camera("http://192.168.1.X:4747/video")
    """
    cap = cv2.VideoCapture(index)

    # بررسی اینکه آیا دوربین باز شد یا نه
    if not cap.isOpened():
        raise IOError(f"دوربین با index={index} پیدا نشد. index دیگه‌ای امتحان کن.")

    # تنظیم رزولوشن تصویر (اختیاریه، حذفش کن اگه مشکل داشت)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"دوربین با موفقیت متصل شد. (index={index})")
    return cap


def read_frame(cap):
    """
    خواندن یک فریم از دوربین

    خروجی:
        ret  → True اگه فریم درست خونده شد، False اگه مشکل داشت
        frame → تصویر به فرمت BGR (فرمت پیش‌فرض cv2)
    """
    ret, frame = cap.read()
    return ret, frame


def release_camera(cap):
    """
    آزاد کردن دوربین و بستن همه پنجره‌های cv2
    همیشه آخر برنامه صداش بزن تا منابع آزاد بشن
    """
    cap.release()
    cv2.destroyAllWindows()
    print("دوربین آزاد شد.")
