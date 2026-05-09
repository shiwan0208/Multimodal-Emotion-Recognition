import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import time

EMOTIONS = ['angry', 'sad', 'happy', 'neutral']
COMPETING = ['angry', 'sad', 'happy', 'neutral']

# Exponential Moving Average alpha. Controls the trade-off between latency and stability.
EMA_ALPHA = 0.6

# Threshold. Filters out noises in model inference output.
NOISE_FLOOR = 0.02       


# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LM_LEFT_LIP = 61
LM_RIGHT_LIP = 291
LM_UPPER_LIP = 13
LM_LOWER_LIP = 14
LM_LEFT_BROW = 52   
LM_LEFT_EYE = 159   
LM_FACE_TOP = 10    
LM_FACE_BOTTOM = 152 


# EMA State
_ema_scores = {k: 0.0 for k in EMOTIONS}
_ema_initialized = False


# Geometric Logic
def get_geometric_ratios(frame):

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # 1. Face Height (Normalization)
    face_h = abs(landmarks[LM_FACE_BOTTOM].y - landmarks[LM_FACE_TOP].y)
    if face_h < 0.01: face_h = 0.1

    # 2. Brow-Eye Ratio (Lower = Angry)
    brow_y = landmarks[LM_LEFT_BROW].y
    eye_y = landmarks[LM_LEFT_EYE].y
    brow_ratio = abs(eye_y - brow_y) / face_h

    # 3. Frown Ratio (Positive = Sad, Negative = Happy)
    corner_y = (landmarks[LM_LEFT_LIP].y + landmarks[LM_RIGHT_LIP].y) / 2.0
    center_y = (landmarks[LM_UPPER_LIP].y + landmarks[LM_LOWER_LIP].y) / 2.0
    frown_ratio = (corner_y - center_y) / face_h

    # BBox for display
    h_min = min([lm.y for lm in landmarks])
    h_max = max([lm.y for lm in landmarks])
    w_min = min([lm.x for lm in landmarks])
    w_max = max([lm.x for lm in landmarks])
    region = {
        'x': int(w_min * w), 'y': int(h_min * h),
        'w': int((w_max - w_min) * w), 'h': int((h_max - h_min) * h)
    }

    return {'brow': brow_ratio, 'frown': frown_ratio}, region


def correct_scores_relative(scores, current, baseline):

    if not baseline or not current:
        return scores

    # Trigger Angry if brow distance drops to 88% of original value
    ANGRY_TRIGGER_RATIO = 0.88 
    
    # Calculate brow change percentage (Current / Base)
    brow_change_pct = current['brow'] / (baseline['brow'] + 1e-6)
    
    # Calculate mouth change (Current - Base)
    frown_diff = current['frown'] - baseline['frown']

    # Display: Current value | Target value (Base * 0.88) | Change rate
    target_brow = baseline['brow'] * ANGRY_TRIGGER_RATIO
    print(f"\rBrow: {current['brow']:.3f} (Trig < {target_brow:.3f}) | Change: {brow_change_pct*100:.1f}%", end='')

    # --- 1. ANGRY CHECK (Relative) ---
    if brow_change_pct < ANGRY_TRIGGER_RATIO:
        # Intensity: The lower below 88%, the higher the bonus
        intensity = (ANGRY_TRIGGER_RATIO - brow_change_pct) * 15 
        bonus = min(intensity, 0.9)
        
        scores['angry'] += bonus
        
        # If brows are lowered, it is absolutely impossible to be Happy
        scores['happy'] = 0.0 
        
        scores['neutral'] = max(0, scores['neutral'] - bonus * 2.0)
        scores['sad'] -= bonus * 0.3

    # --- 2. HAPPY CHECK (Relative) ---
    # Mouth corners moved up relative to baseline (smaller/negative)
    elif frown_diff < -0.015:
        intensity = abs(frown_diff) * 15
        bonus = min(intensity, 0.8)
        
        scores['happy'] += bonus
        scores['sad'] = 0.0
        scores['angry'] = 0.0
        scores['neutral'] = max(0, scores['neutral'] - bonus * 1.5)

    # --- 3. SAD CHECK (Relative) ---
    # Mouth corners moved down relative to baseline (larger/positive)
    elif frown_diff > 0.015:
        intensity = (frown_diff - 0.015) * 20
        bonus = min(intensity, 0.6)
        
        scores['sad'] += bonus
        scores['angry'] -= 0.2
        scores['happy'] = 0.0
        scores['neutral'] = max(0, scores['neutral'] - bonus * 1.5)

    # Normalize
    for k in scores:
        scores[k] = max(0.0, min(1.0, scores[k]))
        
    return scores


# DeepFace Analysis
def analyze_frame(frame, baseline_metrics=None, detector_backend='opencv'):
    global _ema_initialized, _ema_scores

    try:
        # 1. Geometry (Fast & Crucial)
        geo_metrics, region = get_geometric_ratios(frame)
        
        if not geo_metrics:
            return None, None

        # 2. DeepFace (Texture)
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend=detector_backend,
            silent=True
        )

        if not results:
            return None, None

        face = results[0]
        raw = {k: face['emotion'][k] / 100.0 for k in EMOTIONS}

        # 3. Apply Relative Correction
        if baseline_metrics:
            corrected_raw = correct_scores_relative(raw, geo_metrics, baseline_metrics)
        else:
            corrected_raw = raw 

        # 4. EMA Smoothing
        if not _ema_initialized:
            _ema_scores = corrected_raw.copy()
            _ema_initialized = True
        else:
            for k in EMOTIONS:
                _ema_scores[k] = (
                    EMA_ALPHA * corrected_raw[k] +
                    (1 - EMA_ALPHA) * _ema_scores[k]
                )

        return _ema_scores.copy(), region

    except Exception as e:
        return None, None



# Visual Delta Calculation
def calculate_visual_deltas(current, baseline):
    if not current or not baseline:
        return {k: 0.0 for k in EMOTIONS}

    deltas = {}
    for k in EMOTIONS:
        diff = current[k] - baseline.get(k, 0.0)
        deltas[k] = diff if diff > NOISE_FLOOR else 0.0

    return deltas



# Drawing Utility
def draw_face_box(canvas, region, color=(0, 255, 0)):
    if not region:
        return
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)


# Main Execution
def run_visual_demo():
    print("=== Visual Emotion (Personalized Calibration) ===")
    print("Opening Camera...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State: 0=Wait, 1=Calibrate, 2=Ready, 3=Record, 4=Result
    state = 0
    start_time = 0
    duration = 5.0
    
    score_buffer = []      
    baseline_scores = {}   
    
    # Store geometric baseline (The key to accuracy)
    baseline_geo = {'brow': 0.0, 'frown': 0.0}
    geo_buffer = []

    final_result_text = ""
    final_color = (200, 200, 200)

    print(">>> System Ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # 1. Analyze Frame
        # Only pass baseline_geo if we are in testing phase (State >= 2)
        current_base_geo = baseline_geo if state >= 2 else None
        scores, region = analyze_frame(frame, baseline_metrics=current_base_geo, detector_backend='opencv')
        
        # Keep getting raw geo for calibration phase
        curr_geo, _ = get_geometric_ratios(frame)

        # 2. Draw Face Box
        if region:
            draw_face_box(frame, region, color=(0, 255, 0))

        
        if state == 0: # Wait
            cv2.putText(frame, "STEP 1: Calibration", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Stay NEUTRAL -> Press ENTER", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if cv2.waitKey(1) == 13: # Enter
                state = 1
                start_time = time.time()
                score_buffer = []
                geo_buffer = [] 
                print("\n\n[System] Recording Baseline...")

        elif state == 1: # Calibrating
            elapsed = time.time() - start_time
            if scores and curr_geo:
                score_buffer.append(scores)
                geo_buffer.append(curr_geo)
            
            bar_len = int((elapsed / duration) * 200)
            cv2.rectangle(frame, (20, 140), (20 + bar_len, 150), (0, 255, 255), -1)
            cv2.putText(frame, "CALIBRATING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if elapsed >= duration:
                if score_buffer and geo_buffer:
                    # Calc Score Baseline
                    baseline_scores = {k: 0.0 for k in EMOTIONS}
                    for k in EMOTIONS:
                        baseline_scores[k] = np.mean([s[k] for s in score_buffer])
                    
                    # Calc Geometric Baseline (Average of the 5 seconds)
                    avg_brow = np.mean([g['brow'] for g in geo_buffer])
                    avg_frown = np.mean([g['frown'] for g in geo_buffer])
                    baseline_geo = {'brow': avg_brow, 'frown': avg_frown}
                    
                    print(f"\n\n[System] Baseline Established!")
                    print(f"  > Avg Brow Dist: {avg_brow:.4f}")
                    print(f"  > Avg Frown: {avg_frown:.4f}")
                    print(f"  > Anger Trigger Point: < {avg_brow * 0.88:.4f}")
                    
                    state = 2
                else:
                    print("\n[Error] No face detected.")
                    state = 0 

        elif state == 2: # Ready
            cv2.putText(frame, "STEP 2: Ready", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press ENTER to Record", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if cv2.waitKey(1) == 13: # Enter
                state = 3
                start_time = time.time()
                score_buffer = []
                final_result_text = "" 
                print("\n\n[System] Recording Test Emotion...")

        elif state == 3: # Recording
            elapsed = time.time() - start_time
            if scores:
                score_buffer.append(scores)
            
            bar_len = int((elapsed / duration) * 200)
            cv2.rectangle(frame, (20, 140), (20 + bar_len, 150), (0, 255, 0), -1)
            cv2.putText(frame, "RECORDING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if elapsed >= duration:
                if score_buffer:
                    avg_current = {k: np.mean([s[k] for s in score_buffer]) for k in EMOTIONS}
                    deltas = calculate_visual_deltas(avg_current, baseline_scores)
                    
                    winner = "NEUTRAL"
                    max_score = -1
                    
                    for k in COMPETING:
                        if deltas[k] > max_score and deltas[k] > 0.05:
                            max_score = deltas[k]
                            winner = k.upper()
                    
                    final_result_text = winner
                    
                    if winner == "ANGRY": final_color = (0, 0, 255)
                    elif winner == "SAD": final_color = (255, 0, 0)
                    elif winner == "HAPPY": final_color = (0, 255, 0)
                    else: final_color = (200, 200, 200)
                    
                    print(f"\n\n=== TEST RESULT: {winner} ===")
                    print("Detailed Delta Scores:")
                    for k, v in deltas.items():
                        print(f"  {k.upper()}: {v:.4f}")
                    print("===========================\n")
                    
                    state = 4 
                else:
                    state = 2 

        elif state == 4: # Result
            cv2.putText(frame, "RESULT:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, final_result_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, final_color, 4)
            
            cv2.putText(frame, "Press ENTER to Retry", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            if cv2.waitKey(1) == 13: # Enter
                state = 2 

        cv2.imshow("Visual Module", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_demo()