import os
import json
import math
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import base64
import cv2
import numpy as np
import mediapipe as mp
import urllib.parse
import time

# ---------------- SPOTIFY CONFIG ----------------
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random

SPOTIFY_CLIENT_ID = "Spotify-ID"
SPOTIFY_CLIENT_SECRET = "Spotify-Secret"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:5000/callback" 
SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state streaming user-read-email user-read-private"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SPOTIFY_SCOPE,
    cache_path=None 
)

GLOBAL_SESSION = {
    "SP_CLIENT": None,
    "DEVICE_ID": None,
    "TOKEN_INFO": None
}

LANGUAGE_PLAYLISTS = {
    "Hindi": {
        "Happy": "1tTXdi6Bp04Pgmam9bSN7W",
        "Sad": "189Sow1xr7R94oSKs4kISc",
        "Angry": "5cwtgqs4L1fX8IKoQebfjJ",
        "Neutral": "1nAFuLv3VOaQ85D7BlVJj5"
    },
    "Kannada": {
        "Happy": "5TvOfluLFudXPAzrc5f0UP",
        "Sad": "6iC16MXtWgBysN3Ae6S5c7",
        "Angry": "5dj7NXRn2Sx7AQF59LbJhs",
        "Neutral": "52MUOSqvcBDzhNCbYCWF2q"
    },
    "Telugu": {
        "Happy": "4jW37umAGFKr2oQRAk5pAe",
        "Sad": "4gyertnXGs2sSeUWOAKUP2",
        "Angry": "6QI6yorpWKr3Q0UBwKhzOo",
        "Neutral": "0mKX1LScAfgWBl618Eevmu"
    },
    "English": {
        "Happy": "0jrlHA5UmxRxJjoykf7qRY",
        "Sad": "6iVkg4nHwqNjYumzcyboqA",
        "Angry": "67STztGl7srSMNn6hVYPFR",
        "Neutral": "5huWTkhjom3e1XErPgkqZq"
    }
}


# ---------------- UTILS & MEDIAPIPE SETUP ----------------
DATA_FILE = "dataset.json"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.urandom(24)

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands 

# Global lists for Hand Landmarks
TIP_INDICES = [mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
               mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
               mp.solutions.hands.HandLandmark.PINKY_TIP]
MCP_INDICES = [mp.solutions.hands.HandLandmark.THUMB_CMC, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP,
               mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP,
               mp.solutions.hands.HandLandmark.PINKY_MCP]

def save_dataset(dataset):
    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f)

def load_dataset():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def euclid_vec(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def safe_div(a, b):
    return a / b if b != 0 else 0.0

# ---------------- FACE FEATURE EXTRACTION ----------------
def compute_metrics_from_landmarks(lm):
    # (Face feature computation remains the same)
    left_outer_eye = lm[33]
    right_outer_eye = lm[263]
    face_width = dist(left_outer_eye, right_outer_eye) or 1e-6

    top_lip = lm[13]
    bottom_lip = lm[14]
    mouth_left = lm[78]
    mouth_right = lm[308]

    left_eye_top = lm[159]
    left_eye_bottom = lm[145]
    right_eye_top = lm[386]
    right_eye_bottom = lm[374]

    inner_brow_left = lm[70]
    inner_brow_right = lm[300]

    mouth_open = safe_div(dist(top_lip, bottom_lip), face_width)
    mouth_width = safe_div(dist(mouth_left, mouth_right), face_width)
    
    mouth_corner_avg_y = (lm[61][1] + lm[291][1]) / 2
    mouth_center_y = (top_lip[1] + bottom_lip[1]) / 2
    mouth_corner_avg = safe_div(mouth_corner_avg_y - mouth_center_y, face_width)

    eye_open = safe_div(
        dist(left_eye_top, left_eye_bottom) + dist(right_eye_top, right_eye_bottom),
        2 * face_width
    )

    brow_height = safe_div(
        abs(inner_brow_left[1] - left_eye_top[1]) + abs(inner_brow_right[1] - right_eye_top[1]),
        2 * face_width
    )

    brow_inner_distance = safe_div(
        dist(inner_brow_left, inner_brow_right), face_width
    )

    vec = [
        round(mouth_width, 4), round(mouth_open, 4), round(mouth_corner_avg, 4),
        round(eye_open, 4), round(brow_height, 4), round(brow_inner_distance, 4)
    ]

    return vec, {}


# ---------------- HAND GESTURE DETECTION (SIMPLIFIED) ----------------
def detect_hand_gesture(hand_landmarks):
    lm_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    
    # Check if a finger is UP (tip y coordinate is significantly less than MCP/wrist y coordinate)
    def is_finger_up(finger_idx):
        if finger_idx == 0: # Thumb: relative to wrist
             return lm_list[TIP_INDICES[0]][1] < lm_list[mp_hands.HandLandmark.WRIST][1] - 0.1
        # Other fingers: relative to MCP
        return lm_list[TIP_INDICES[finger_idx]][1] < lm_list[MCP_INDICES[finger_idx]][1] - 0.05
    
    # Check if a finger is DOWN
    def is_finger_down(finger_idx):
        if finger_idx == 0:
            return lm_list[TIP_INDICES[0]][1] > lm_list[mp_hands.HandLandmark.WRIST][1]
        return lm_list[TIP_INDICES[finger_idx]][1] > lm_list[MCP_INDICES[finger_idx]][1] + 0.05
    
    # --- 1. Closed Fist (✊ Pause) ---
    # Index to Pinky are down (ignoring thumb for robust fist detection)
    is_fist = (is_finger_down(1) and is_finger_down(2) and is_finger_down(3) and is_finger_down(4))
    if is_fist:
        return "PLAY_PAUSE_PAUSE"

    # --- 2. Open Palm (✋ Play) ---
    # Index to Pinky are up
    is_open_palm = (is_finger_up(1) and is_finger_up(2) and is_finger_up(3) and is_finger_up(4))
    if is_open_palm:
        return "PLAY_PAUSE_PLAY"
        
    return "NONE"


# ---------------- landmark extraction ----------------
def extract_landmarks_from_image_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, "Could not decode image"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_lm, hand_lm = None, None

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as fm:
        res = fm.process(img_rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            face_lm = [(p.x, p.y) for p in lm]

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            hand_lm = res.multi_hand_landmarks[0]

    if not face_lm and not hand_lm:
        return None, None, "No face or hand detected"
    
    return face_lm, hand_lm, None


# ---------------- dataset ops & KNN ----------------
def add_sample(label, feature_vec):
    ds = load_dataset()
    ds.append({"label": label, "vector": feature_vec})
    save_dataset(ds)
    return len(ds)

def clear_dataset():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    return True

def knn_predict(vec, k=3):
    ds = load_dataset()
    if not ds:
        return None, "no_samples"

    dists = [(euclid_vec(vec, s["vector"]), s["label"])
             for s in ds]
    dists.sort(key=lambda x: x[0])

    top = dists[:k]

    counts = {}
    for _, label in top:
        counts[label] = counts.get(label, 0) + 1

    best = max(counts, key=counts.get)
    votes = counts[best]
    conf_votes = votes / k

    avg_dist = sum(d for d, _ in top) / k
    conf_dist = max(0, 1 - (avg_dist / 0.25))

    confidence = round(conf_votes * 0.6 + conf_dist * 0.4, 3)

    return {"label": best, "confidence": confidence}, None

# ---------------- FLASK ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

# (Login, Callback, is_logged_in, set_device, play_song remain the same)
@app.route("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route("/callback")
def callback():
    global GLOBAL_SESSION
    
    token_info = sp_oauth.get_access_token(request.args['code'])
    
    GLOBAL_SESSION["TOKEN_INFO"] = token_info
    GLOBAL_SESSION["SP_CLIENT"] = spotipy.Spotify(auth=token_info['access_token'])
    
    return redirect("/")


@app.route("/is_logged_in", methods=["GET"])
def is_logged_in():
    token_info = GLOBAL_SESSION.get("TOKEN_INFO")
    logged_in = GLOBAL_SESSION["SP_CLIENT"] is not None and token_info is not None
    
    if logged_in:
        if token_info and time.time() > token_info.get('expires_at', 0):
             logged_in = False
             GLOBAL_SESSION["SP_CLIENT"] = None
             GLOBAL_SESSION["TOKEN_INFO"] = None

    return jsonify({
        "logged_in": logged_in,
        "access_token": token_info['access_token'] if logged_in else None
    })

@app.route("/set_device", methods=["POST"])
def set_device():
    device_id = request.json.get("device_id")
    if device_id:
        GLOBAL_SESSION["DEVICE_ID"] = device_id
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "No device_id provided"})

@app.route("/play_song", methods=["POST"])
def play_song():
    sp_client = GLOBAL_SESSION["SP_CLIENT"]
    device_id = GLOBAL_SESSION["DEVICE_ID"]
    track_uri = request.json.get("uri")

    if not sp_client or not device_id:
        return jsonify({"ok": False, "error": "User not authenticated or device not set."})

    try:
        sp_client.start_playback(device_id=device_id, uris=[track_uri])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ---------------- GESTURE COMMAND ROUTE (SIMPLIFIED) ----------------
@app.route("/gesture_command", methods=["POST"])
def gesture_command():
    sp_client = GLOBAL_SESSION["SP_CLIENT"]
    device_id = GLOBAL_SESSION["DEVICE_ID"]
    command = request.json.get("command")

    if not sp_client or not device_id:
        return jsonify({"ok": False, "error": "User not authenticated or device not set."})

    try:
        status = ""
        # Only handle PLAY and PAUSE
        if command in ("PLAY_PAUSE_PAUSE", "PLAY_PAUSE_PLAY"):
            current = sp_client.current_playback()
            is_playing = current and current.get('is_playing')
            
            if command == "PLAY_PAUSE_PAUSE" and is_playing:
                sp_client.pause_playback(device_id=device_id)
                status = "Playback paused (✊)"
            elif command == "PLAY_PAUSE_PLAY" and not is_playing:
                sp_client.start_playback(device_id=device_id)
                status = "Playback resumed (✋)"
            else:
                 status = "Playback state unchanged"
            
        else:
            return jsonify({"ok": False, "error": "Invalid command (Only Play/Pause supported)"})
        
        return jsonify({"ok": True, "status": status})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# (Predict, Songs, Dataset, Reset routes remain the same)
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    data = payload.get("image")

    img_bytes = base64.b64decode(data.split(",")[1])
    face_lm, hand_lm, err = extract_landmarks_from_image_bytes(img_bytes)

    if err:
        return jsonify({"ok": False, "error": err})

    emotion_result = None
    if face_lm:
        vec, _ = compute_metrics_from_landmarks(face_lm)
        emotion_result, _ = knn_predict(vec)
    
    gesture_command_result = "NONE"
    if hand_lm:
        gesture_command_result = detect_hand_gesture(hand_lm)

    final_emotion = emotion_result["label"] if emotion_result else "Neutral"
    final_confidence = emotion_result["confidence"] if emotion_result else 0.0

    return jsonify({
        "ok": True,
        "prediction": final_emotion,
        "confidence": final_confidence,
        "gesture": gesture_command_result
    })

@app.route("/songs", methods=["POST"])
def songs():
    sp_client = GLOBAL_SESSION["SP_CLIENT"]
    if sp_client is None:
        return jsonify({"ok": False, "error": "user_not_authenticated"})

    payload = request.json
    emotion = payload.get("emotion")
    language = payload.get("language")

    playlist_id = LANGUAGE_PLAYLISTS.get(language, {}).get(emotion)
    if not playlist_id:
        return jsonify({"ok": False, "error": "playlist_not_found"})

    try:
        results = sp_client.playlist_tracks(playlist_id, limit=20) 
        tracks = []
        for item in results.get("items", []):
            track = item.get("track")
            if track:
                tracks.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "uri": track["uri"]
                })
        return jsonify({"ok": True, "songs": tracks[:4]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/dataset", methods=["GET"])
def dataset():
    ds = load_dataset()
    count = len(ds)

    by = {}
    for s in ds:
        by[s["label"]] = by.get(s["label"], 0) + 1

    return jsonify({"count": count, "by_label": by})


@app.route("/reset", methods=["POST"])
def reset():
    clear_dataset()
    return jsonify({"ok": True})


@app.route("/static/<path:path>")
def staticfile(path):
    return send_from_directory("static", path)


if __name__ == "__main__":

    app.run(debug=True)
