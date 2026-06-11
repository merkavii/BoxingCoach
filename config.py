
# ------------------- تنظیمات دوربین -------------------
CAMERA_INDEX = 0  # اگه DroidCam شناخته نشد، 1 یا 2 امتحان کن
DROIDCAM_IP = "http://192.168.1.X:4747/video"  # اگه با IP وصل میشی اینو عوض کن

# ------------------- تنظیمات mediapipe -------------------
NUM_KEYPOINTS = 33   # تعداد نقاط بدن که mediapipe برمیگردونه
KEYPOINT_DIM = 3     # هر نقطه 3 مقدار داره: x, y, z
INPUT_SIZE = NUM_KEYPOINTS * KEYPOINT_DIM  # = 99، ورودی مدل LSTM

# ------------------- تنظیمات مدل -------------------
SEQUENCE_LENGTH = 30  # تعداد فریم‌هایی که برای هر حرکت در نظر میگیریم

# لیست حرکاتی که مدل باید یاد بگیره تشخیص بده
ACTIONS = ['jab', 'cross', 'hook', 'uppercut', 'guard', 'footwork']

# ------------------- مسیر فایل‌ها -------------------
DATA_RAW_PATH = "data/raw/dataset.csv"
DATA_NPY_X = "data/processed/X.npy"
DATA_NPY_Y = "data/processed/y.npy"
MODEL_SAVE_PATH = "models/saved/boxing_lstm_v1.pth"
