import os
import sys
import numpy as np

PRIMARY_MODEL = os.path.join(os.path.dirname(__file__), "sig_bak_ovr.onnx")
P808_MODEL = os.path.join(os.path.dirname(__file__), "model_v8.onnx")

SR = 16000
INPUT_LENGTH = 9.01


def audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000):
    import librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def get_polyfit_val(sig, bak, ovr):
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


def compute_dnsmos(audio, primary_sess, p808_sess, sr=SR):
    len_samples = int(INPUT_LENGTH * sr)
    while len(audio) < len_samples:
        audio = np.concatenate([audio, audio])
    audio = audio[:len_samples]

    oi = primary_sess.get_inputs()[0].name
    out = primary_sess.run(None, {oi: audio[np.newaxis, :]})[0][0]
    sig, bak, ovr = get_polyfit_val(out[0], out[1], out[2])

    p808_input = np.array(audio_melspec(audio=audio[:-160])).astype('float32')[np.newaxis, :, :]
    oi808 = p808_sess.get_inputs()[0].name
    p808_mos = p808_sess.run(None, {oi808: p808_input})[0][0][0]

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
