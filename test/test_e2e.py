"""Real-upscale end-to-end test.

Skipped cleanly when audiosronnx or its weights are unavailable (e.g. offline
CI), so it never blocks the suite while still verifying a genuine 48 kHz upscale
when the model can be fetched.
"""
import os
import tempfile

import numpy as np
import pytest

from ovos_tts_transformer_audiosr import AudioSRTTSTransformer


def _real_engine_or_skip(engine="novasr"):
    try:
        from audiosronnx import load_sr
    except ImportError:
        pytest.skip("audiosronnx not installed")
    try:
        return load_sr(engine=engine)
    except Exception as exc:  # weights unavailable / offline
        pytest.skip(f"audiosronnx '{engine}' weights unavailable: {exc}")


def test_real_upscale_16k_to_48k():
    engine = _real_engine_or_skip("novasr")
    from audiosronnx.audio import read_wav, write_wav

    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        sr_in = 16000
        t = np.linspace(0, 0.5, int(sr_in * 0.5), endpoint=False)
        audio = 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        write_wav(wav, audio, sr_in)

        tx = AudioSRTTSTransformer(config={"engine": "novasr"})
        tx._sr_engine = engine
        tx._sr_loaded = True

        out_path, _ = tx.transform(wav)
        assert out_path.endswith("_sr.wav")
        out, out_sr = read_wav(out_path)
        assert out_sr == 48000
        assert out.size > 0
        assert float(np.max(np.abs(out))) > 1e-3
