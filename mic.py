import sounddevice as sd
import numpy as np
import librosa

fs = 16000


# Utility: Compute P80 Energy + frame energies
def p80_energy(y, frame=256, hop=128):
    energies = []
    for i in range(0, len(y) - frame, hop):
        frame_y = y[i:i+frame]
        e = np.sqrt(np.mean(frame_y**2))
        energies.append(e)
    energies = np.array(energies) if len(energies) else np.array([0.0])
    p80 = np.percentile(energies, 80)
    return p80, energies


# Extract All Features (v7.5)
def extract_features(y):
    y = y.astype(float)

    # Pitch 
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=350, sr=fs)
    f0 = f0[~np.isnan(f0)]
    if len(f0):
        pitch = np.mean(f0)
        pitch_var = np.std(f0)
    else:
        pitch = 0.0
        pitch_var = 0.0

    # Energy P80 + energy variance
    energy_p80, energy_frames = p80_energy(y)
    energy_var = float(np.std(energy_frames)) if len(energy_frames) else 0.0

    # Spectral Centroid (Brightness) 
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs)[0].mean()

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)
    mfcc_std = np.std(mfcc, axis=1).mean()

    # Delta
    delta = librosa.feature.delta(mfcc)
    delta_std = np.std(delta, axis=1).mean()

    # Spectral Flux
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    if S.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2)) / S.shape[1]
    else:
        flux = 0.0

    # High/Low energy 
    freqs = librosa.fft_frequencies(sr=fs, n_fft=1024)
    low = S[freqs < 1500].sum()
    high = S[freqs >= 1500].sum()

    # HNR
    harm = librosa.effects.harmonic(y)
    noise = y - harm
    harm_e = np.sum(harm**2)
    noise_e = np.sum(noise**2) + 1e-6
    hnr = harm_e / noise_e

    # MFCC slope
    coeffs = mfcc.mean(axis=1)
    slope = coeffs[1] - coeffs[-1]

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()

    return (
        pitch, energy_p80, centroid, mfcc_std, delta_std,
        flux, hnr, slope, low, high,
        pitch_var, energy_var, zcr
    )


# Classifier 
def classify_v7_5(
    rel_pitch, rel_energy, rel_flux, rel_centroid,
    rel_hnr, rel_slope, rel_mfcc,
    rel_pitch_var, rel_energy_var, rel_zcr,
    low, high
):

    print("\nSCORE :")


    # 1. AROUSAL
    arousal_score = 0
    if rel_pitch > 1.15: 
        print("Arousal +1 (Pitch ↑)")
        arousal_score += 1
    if rel_energy > 1.20: 
        print("Arousal +1 (Energy ↑)")
        arousal_score += 1.5 
    if rel_flux > 1.5: 
        print("Arousal +1 (Flux ↑)")
        arousal_score += 1

    arousal = "HIGH" if (arousal_score >= 1.5) else "LOW"
    print(f"Arousal Score = {arousal_score} → {arousal}\n")


    # 2. VALENCE (Positive/Negative)
    valence_score = 0

    if rel_centroid > 1.05:
        valence_score += 1; print("Valence +1 (Centroid ↑)")

    if (high / (low + 1e-6)) > 0.90:
        valence_score += 1; print("Valence +1 (High/Low ↑)")

    if rel_hnr > 0.85:
        valence_score += 1; print("Valence +1 (HNR ↑)")

    # Flux is usually high for laughter, so only deduct if extreme
    # Only consider it calm (Positive) if Flux is relatively low
    if rel_flux < 3.5: 
        valence_score += 1; print("Valence +1 (Flux not extreme)")

    if rel_slope > 1.00:
        valence_score += 1; print("Valence +1 (MFCC Slope ↑)")

    # Happy Feature: High pitch variance and high pitch -> Positive
    if rel_pitch_var > 1.2 and rel_pitch > 1.1:
        valence_score += 1.5; print("Valence +1.5 (Bouncy & High Pitch)")

    valence = "POSITIVE" if valence_score >= 3 else "NEGATIVE"
    print(f"Valence Score = {valence_score} → {valence}\n")


    # 3. HAPPY (Laugh) SCORE
    laugh_score = 0
    
    # Laughter: Bright, Chaotic, Breathy, Trembling
    if rel_centroid > 1.10:
        laugh_score += 1; print("Laugh +1 (Centroid ↑)")
    if 2.0 < rel_flux < 8.0:
        laugh_score += 1; print("Laugh +1 (Flux moderate)")
    if rel_hnr < 0.75:
        laugh_score += 1; print("Laugh +1 (HNR ↓)")
    if rel_zcr > 1.2:
        laugh_score += 1; print("Laugh +1 (ZCR high)")
    
    # Laughter is "bouncy" (He-He-He)
    if rel_pitch_var > 1.5:
        laugh_score += 2; print("Laugh +2 (Very Bouncy/Pitch Var ↑↑)")
    
    # [If pitch is high and valence is positive
    if rel_pitch > 1.2 and valence == "POSITIVE":
        laugh_score += 1; print("Laugh +1 (High Pitch Positive)")

    print(f"Laugh Score = {laugh_score}\n")


    # 4. SADNESS SCORE (Low Arousal)
    sadness_score = 0
    if rel_pitch_var < 0.85:
        sadness_score += 1; print("Sad +1 (Pitch variance ↓)")
    if rel_energy < 0.8:
        sadness_score += 1; print("Sad +1 (Energy ↓)")
    if rel_energy_var < 0.85:
        sadness_score += 1; print("Sad +1 (Energy variance ↓)")
    if rel_zcr < 0.85:
        sadness_score += 1; print("Sad +1 (ZCR ↓)")

    print(f"Sadness Score = {sadness_score}\n")


    # 5. CRY SCORE (High Arousal Sad)
    cry_score = 0
    if rel_pitch_var > 2.0:
        cry_score += 2; print("Cry +1 (Pitch variance ↑↑)")
    if rel_centroid < 0.95:
        cry_score += 1.5; print("Cry +1.5 (Centroid ↓)") 
    if 1.5 < rel_flux < 5.0:
        cry_score += 1; print("Cry +1 (Flux moderate)")
    if rel_centroid > 1.15:
        cry_score -= 1; print("Cry -1 (Too Bright)")

    print(f"Cry Score = {cry_score}\n")


    # 6. ANGRY SCORE
    angry_score = 0

    if rel_hnr < 0.65:
        angry_score += 1; print("Angry +1 (HNR low)")
    if rel_flux > 4.5:
        angry_score += 1; print("Angry +1 (Flux chaotic)")
    if rel_zcr > 1.6:
        angry_score += 1; print("Angry +1 (ZCR high)")
    if rel_energy_var > 3.0:
        angry_score += 1; print("Angry +1 (Energy variance high)")
    if rel_energy > 1.6:
        angry_score += 1; print("Angry +1 (Very Loud)")
    if rel_centroid > 1.1:
        angry_score += 1; print("Angry +1 (Sharp Tone)")

    # Too bouncy -> Likely Laugh or Cry, not sustained Anger
    if rel_pitch_var > 1.5:
        angry_score -= 2; print("Angry -2 (Too Bouncy -> Likely Laugh/Cry)")

    # High pitch and Positive -> Likely Happy scream
    if rel_pitch > 1.3 and valence == "POSITIVE":
        angry_score -= 2; print("Angry -2 (High Pitch & Positive -> Likely Happy)")

    print(f"Angry Score = {angry_score}\n")

    
    # 7. FINAL DECISION  
    emotion = "Neutral"

    if arousal == "HIGH":
        
        scores = {'Angry': angry_score, 'Sad': cry_score, 'Happy': laugh_score}
        winner = max(scores, key=scores.get)
        max_val = scores[winner]
        
        print(f"Competition: {scores}, Winner: {winner}")

        # Valence Decider
        # If Happy and Angry are close, check Valence
        if abs(angry_score - laugh_score) <= 2.0:
            if valence == "POSITIVE":
                print("[Tie-Breaker] Valence Positive -> Shift to Happy")
                winner = 'Happy'
                max_val = max(max_val, laugh_score)
            else:
                pass 

        # Timbre Decider (Angry vs Sad)
        if winner in ['Angry', 'Sad'] and abs(angry_score - cry_score) <= 1.5:
            if rel_centroid < 1.0:
                print("[Tie-Breaker] Centroid Low -> Shift to Sad")
                winner = 'Sad'
                max_val = max(max_val, cry_score)
            elif rel_centroid > 1.1:
                print("[Tie-Breaker] Centroid High -> Shift to Angry")
                winner = 'Angry'
                max_val = max(max_val, angry_score)

        # Threshold check
        threshold = 1.0 
        if max_val >= threshold:
            emotion = winner
        else:
            emotion = "Neutral"
    
    else:   # arousal == LOW
        if sadness_score >= 2:
            emotion = "Sad"
        else:
            emotion = "Neutral"

    raw_scores = {
        'laugh': laugh_score,
        'sadness': sadness_score,
        'cry': cry_score,
        'angry': angry_score
    }

    return emotion, arousal, valence, raw_scores


# Recording
def record(sec=5):
    print(f"Recording... {sec} seconds")
    y = sd.rec(int(sec * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete\n")
    return y.flatten()


def compute_baseline(y):
    """
    Called by main.py to get baseline features.
    Returns: (features_array, None)
    """
    feats = extract_features(y)
    return feats, None 

def audio_emotion_scores(y, base_feats, _unused_std=None):

    # 1. Extract
    curr_feats = extract_features(y)
    
    # 2. Convert to numpy for calculation
    curr = np.array(curr_feats)
    base = np.array(base_feats)
    
    # Calculate Ratios
    rel = curr / (base + 1e-6)
    
    # Unpack for Classifier (Indices match extract_features return)
    # 0:pitch, 1:energy, 2:centroid, 3:mfcc, 4:delta, 5:flux, 
    # 6:hnr, 7:slope, 8:low, 9:high, 10:pvar, 11:evar, 12:zcr
    
    rel_pitch      = rel[0]
    rel_energy     = rel[1]
    rel_centroid   = rel[2]
    rel_mfcc       = rel[3]
    rel_flux       = rel[5]
    rel_hnr        = rel[6]
    rel_slope      = rel[7]
    low_raw        = curr[8]
    high_raw       = curr[9]
    rel_pvar       = rel[10]
    rel_evar       = rel[11]
    rel_zcr        = rel[12]
    
    # 3. Classify
    emotion, arousal, valence, raw = classify_v7_5(
        rel_pitch, rel_energy, rel_flux, rel_centroid,
        rel_hnr, rel_slope, rel_mfcc,
        rel_pvar, rel_evar, rel_zcr,
        low_raw, high_raw
    )
    
    # 4. Normalize Scores (0.0 - 1.0)
    # Estimated max scores: Angry~6, Laugh~6, Sad/Cry~5
    # We use conservative divisors to allow hitting 1.0 easily
    
    angry_norm = min(1.0, max(0, raw['angry']) / 5.0)
    happy_norm = min(1.0, max(0, raw['laugh']) / 5.0)
    
    # Sad is max of 'sadness' (Low Arousal) or 'cry' (High Arousal)
    sad_norm_low = max(0, raw['sadness']) / 3.0
    sad_norm_high = max(0, raw['cry']) / 5.0
    sad_norm = min(1.0, max(sad_norm_low, sad_norm_high))
    
    # Neutral is calculated by residual
    max_active = max(angry_norm, happy_norm, sad_norm)
    neutral_norm = max(0.0, 1.0 - max_active)
    
    return {
        'angry': angry_norm,
        'sad': sad_norm,
        'happy': happy_norm,
        'neutral': neutral_norm
    }


if __name__ == "__main__":
    
    def run_system():
        print("Recording Baseline (Neutral) for 5 seconds")
        input("Press Enter to start")
        yb = record(5)
        yb, _ = librosa.effects.trim(yb)

        base_feats, _ = compute_baseline(yb)
        
        # Just to show baseline features
        (
            b_pitch, b_energy, b_centroid, b_mfcc, b_delta, b_flux,
            b_hnr, b_slope, b_low, b_high,
            b_pvar, b_evar, b_zcr
        ) = base_feats

        print("\nBaseline Features :")
        print(f"Pitch        = {b_pitch:.2f}")
        print(f"Energy       = {b_energy:.5f}")
        print(f"Flux         = {b_flux:.2f}")
        print(f"Centroid     = {b_centroid:.2f}")
        print(f"HNR          = {b_hnr:.2f}")
        print(f"Slope        = {b_slope:.2f}")
        print(f"Pitch_var    = {b_pvar:.2f}")
        print(f"Energy_var   = {b_evar:.2f}")
        print(f"ZCR          = {b_zcr:.2f}")


        input("\nPress Enter to start emotion recording (5 seconds)")
        y = record(5)
        y, _ = librosa.effects.trim(y)
        
        # Test the full pipeline
        scores = audio_emotion_scores(y, base_feats)
        print("\nFinal Normalized Scores:", scores)

    run_system()