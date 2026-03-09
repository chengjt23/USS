import os
import sys
import numpy as np

PRIMARY_MODEL = os.path.join(os.path.dirname(__file__), "sig_bak_ovr.onnx")
P808_MODEL = os.path.join(os.path.dirname(__file__), "model_v8.onnx")

SR = 16000
INPUT_LENGTH = 9.01
N_FFT = 320
HOP_LENGTH = 160
N_MELS = 120


def audio_melspec(audio, sr=SR):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)


def get_polyfit_val(sig, bak, ovr):
    p_sig = np.poly1d([-0.08572166, 0.40221407, 1.0839862, -0.22786875])
    p_bak = np.poly1d([-0.13543826, 0.87185817, 0.19576773, 0.13120761])
    p_ovr = np.poly1d([-0.03255527, 0.23521211, -0.18003412, 0.92045015])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


def compute_dnsmos(audio, primary_sess, p808_sess, sr=SR):
    len_samples = int(INPUT_LENGTH * sr)
    while len(audio) < len_samples:
        audio = np.concatenate([audio, audio])
    audio = audio[:len_samples]

    mel = audio_melspec(audio, sr=sr)
    p808_mel = audio_melspec(audio[:int(sr * 9)], sr=sr)

    oi = primary_sess.get_inputs()[0].name
    out = primary_sess.run(None, {oi: mel})[0]
    sig, bak, ovr = get_polyfit_val(out[0], out[1], out[2])

    oi808 = p808_sess.get_inputs()[0].name
    p808_mos = p808_sess.run(None, {oi808: p808_mel})[0][0]

    return {"SIG": float(sig), "BAK": float(bak), "OVR": float(ovr), "P808_MOS": float(p808_mos)}


if __name__ == "__main__":
    for path in [PRIMARY_MODEL, P808_MODEL]:
        if not os.path.exists(path):
            print(f"ERROR: model not found: {path}")
            sys.exit(1)

    import onnxruntime as ort
    primary_sess = ort.InferenceSession(PRIMARY_MODEL, providers=["CPUExecutionProvider"])
    p808_sess = ort.InferenceSession(P808_MODEL, providers=["CPUExecutionProvider"])
    print("Models loaded.")

    rng = np.random.default_rng(42)
    t = np.linspace(0, 5, SR * 5, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    noise = rng.standard_normal(SR * 5).astype(np.float32) * 0.1
    mixed = (sine + rng.standard_normal(SR * 5).astype(np.float32) * 0.3).astype(np.float32)

    for name, audio in [("white noise", noise), ("sine wave", sine), ("sine + noise", mixed)]:
        scores = compute_dnsmos(audio, primary_sess, p808_sess)
        print(f"\n[{name}]  " + "  ".join(f"{k}: {v:.3f}" for k, v in scores.items()))
