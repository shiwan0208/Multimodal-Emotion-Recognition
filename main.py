import cv2
import time
import threading
import numpy as np
import librosa
import math

import mic
import visual


WINDOW_DURATION = 5.0
VISUAL_WEIGHT = 0.5
AUDIO_WEIGHT  = 0.5
SOFTMAX_TEMP  = 0.8       
EMOTIONS = ['angry', 'sad', 'happy', 'neutral']


def softmax(scores, temp=1.0):
    exps = [math.exp(s / temp) for s in scores]
    s = sum(exps) + 1e-9
    return [e / s for e in exps]


class AudioThread:
    def __init__(self):
        self.done = False
        self.base_feats = None # Stores raw audio features
        self.scores = {'angry':0,'sad':0,'happy':0,'neutral':1}

    def _task(self, duration, baseline=False):
        self.done = False
        
        # Call mic.py to record
        y = mic.record(duration)
        y, _ = librosa.effects.trim(y)

        if baseline:
            self.base_feats, _ = mic.compute_baseline(y)
            print(">>> [Audio] Baseline captured.")
        else:
            # Pass captured baseline features
            self.scores = mic.audio_emotion_scores(
                y, self.base_feats
            )

        self.done = True

    def start(self, duration, baseline=False):
        t = threading.Thread(
            target=self._task,
            args=(duration, baseline)
        )
        t.start()


def main():
    cap = cv2.VideoCapture(0)
    audio = AudioThread()

    state = 0
    start_time = 0

    # Visual Buffers
    visual_score_buffer = []
    visual_geo_buffer = []  # To store geometry for baseline calculation
    
    # Baselines
    visual_baseline_scores = {}
    visual_baseline_geo = None # Stores {'brow': x, 'frown': y}

    final_result_text = "Waiting..."
    final_result_color = (200, 200, 200)

    print("=== Integrated Multimodal System ===")
    print("Supports: Relative Geometry Calibration & Audio Analysis")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Canvas setup
        viz_w = 400 
        canvas = np.zeros((h, w + viz_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        canvas[:, w:] = (30, 30, 30)

        key = cv2.waitKey(1)

        # ---------------- Visual Analysis ----------------
        # In state 3 (Recording), we pass the geometric baseline to enable relative correction
        current_base_geo = visual_baseline_geo if state == 3 else None
        
        v_scores, face = visual.analyze_frame(frame, baseline_metrics=current_base_geo)
        
        # Also get raw geometry for calibration phase
        curr_geo, _ = visual.get_geometric_ratios(frame)

        if face:
            visual.draw_face_box(canvas, face)


        if state == 0:  # WAIT
            cv2.putText(canvas, "STEP 1: Calibration", (w+20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(canvas, "Stay Neutral & Silent", (w+20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(canvas, "Press [ENTER]", (w+20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if key == 13: # Enter
                visual_score_buffer = []
                visual_geo_buffer = []
                start_time = time.time()
                audio.start(WINDOW_DURATION, baseline=True)
                state = 1

        elif state == 1:  # CALIBRATING
            elapsed = time.time() - start_time
            
            # Buffer Data
            if v_scores: visual_score_buffer.append(v_scores)
            if curr_geo: visual_geo_buffer.append(curr_geo)

            # UI Progress
            bar = int((elapsed / WINDOW_DURATION) * viz_w)
            cv2.rectangle(canvas, (w, 0), (w + bar, 10), (0,255,255), -1)
            cv2.putText(canvas, "CALIBRATING...", (w+20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if elapsed >= WINDOW_DURATION and audio.done:
                # 1. Compute Score Baseline
                if visual_score_buffer:
                    visual_baseline_scores = {
                        k: np.mean([v[k] for v in visual_score_buffer])
                        for k in visual.EMOTIONS 
                    }
                else:
                    visual_baseline_scores = {k: 0.0 for k in visual.EMOTIONS}
                
                # 2. Compute Geometric Baseline (Avg of 5s)
                if visual_geo_buffer:
                    avg_brow = np.mean([g['brow'] for g in visual_geo_buffer])
                    avg_frown = np.mean([g['frown'] for g in visual_geo_buffer])
                    visual_baseline_geo = {'brow': avg_brow, 'frown': avg_frown}
                    
                    print(f"\n[Main] Visual Baseline Geo: Brow={avg_brow:.3f}, Frown={avg_frown:.3f}")
                    print(f"[Main] Angry Trigger: < {avg_brow * 0.88:.3f}")
                
                print(">>> Calibration Done.")
                state = 2

        elif state == 2:  # READY
            cv2.putText(canvas, "STEP 2: Ready", (w+20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(canvas, "Press [ENTER] to Record", (w+20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.putText(canvas, "PREDICTION:", (w+20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            cv2.putText(canvas, final_result_text, (w+20, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, final_result_color, 3)

            if key == 13:
                visual_score_buffer = []
                start_time = time.time()
                audio.start(WINDOW_DURATION, baseline=False)
                state = 3

        elif state == 3:  # RECORDING
            elapsed = time.time() - start_time
            
            if v_scores: visual_score_buffer.append(v_scores)

            bar = int((elapsed / WINDOW_DURATION) * viz_w)
            cv2.rectangle(canvas, (w, 0), (w + bar, 10), (0,255,0), -1)
            cv2.putText(canvas, "RECORDING...", (w+20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if elapsed >= WINDOW_DURATION and audio.done:
                print("\n=== FINAL FUSION RESULT ===")
                
                # 1. Visual Delta Calculation
                if visual_score_buffer:
                    avg_visual = {k: np.mean([v[k] for v in visual_score_buffer]) for k in visual.EMOTIONS}
                    v_delta = visual.calculate_visual_deltas(avg_visual, visual_baseline_scores)
                else:
                    v_delta = {k:0 for k in visual.EMOTIONS}
                
                print(f"Visual Delta: {v_delta}")

                # 2. Audio Scores
                a_scores = audio.scores
                print(f"Audio Scores: {a_scores}")

                # 3. Fusion
                fused = {}
                for e in EMOTIONS:
                    v_val = v_delta.get(e, 0.0)
                    a_val = a_scores.get(e, 0.0)
                    fused[e] = (VISUAL_WEIGHT * v_val + AUDIO_WEIGHT * a_val)

                print(f"Fused Scores: {fused}")

                # 4. Determine Winner
                probs = softmax([fused[e] for e in EMOTIONS], SOFTMAX_TEMP)
                best_i = int(np.argmax(probs))
                best_emotion = EMOTIONS[best_i]
                
                if best_emotion == "neutral":
                    final_result_text = "NEUTRAL"
                    final_result_color = (200, 200, 200)
                else:
                    final_result_text = best_emotion.upper()
                    if final_result_text == "ANGRY": final_result_color = (0, 0, 255)
                    elif final_result_text == "SAD": final_result_color = (255, 0, 0)
                    elif final_result_text == "HAPPY": final_result_color = (0, 255, 0)
                    else: final_result_color = (200, 200, 200)

                print(f"WINNER: {final_result_text}\n")
                state = 2

        cv2.imshow("Multimodal Emotion System", canvas)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()