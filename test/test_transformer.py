"""Unit tests for the audiosr TTS transformer.

The audiosronnx super-resolution engine is mocked so these tests never download
weights or run inference — they exercise the transformer's wiring, config
handling and graceful degradation.
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ovos_tts_transformer_audiosr import AudioSRTTSTransformer

# read_wav/write_wav are pure DSP (no network), so real ones are used to build
# and inspect fixtures.
from audiosronnx.audio import read_wav, write_wav


def _make_wav(path, sample_rate=16000, seconds=0.1):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    write_wav(path, audio, sample_rate)
    return audio


def _fake_engine():
    """An engine that 'upscales' by naive 3x repeat to 48 kHz."""
    eng = MagicMock()

    def upscale(audio, in_sr):
        return np.repeat(np.asarray(audio, dtype=np.float32), 3), 48000

    eng.upscale.side_effect = upscale
    return eng


def test_default_engine_is_novasr():
    tx = AudioSRTTSTransformer()
    assert tx.engine_name == "novasr"


def test_engine_from_config():
    tx = AudioSRTTSTransformer(config={"engine": "lavasr"})
    assert tx.engine_name == "lavasr"


def test_entrypoint_registered():
    from ovos_plugin_manager.tts_transformers import find_tts_transformer_plugins
    plugins = find_tts_transformer_plugins()
    assert "ovos-tts-transformer-audiosr" in plugins


def test_transform_upscales_to_48k():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        _make_wav(wav, sample_rate=16000)
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", return_value=_fake_engine()) as loader:
            out_path, ctx = tx.transform(wav)
        loader.assert_called_once_with(engine="novasr")
        assert out_path != wav
        assert out_path.endswith("_sr.wav")
        _, out_sr = read_wav(out_path)
        assert out_sr == 48000


def test_engine_is_cached_across_calls():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        _make_wav(wav, sample_rate=16000)
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", return_value=_fake_engine()) as loader:
            tx.transform(wav)
            tx.transform(wav)
        loader.assert_called_once()  # loaded once, cached thereafter


def test_already_48k_is_passed_through():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "hires.wav")
        _make_wav(wav, sample_rate=48000)
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", return_value=_fake_engine()) as loader:
            out_path, ctx = tx.transform(wav)
        assert out_path == wav  # untouched


def test_load_failure_degrades_to_passthrough():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        _make_wav(wav, sample_rate=16000)
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", side_effect=RuntimeError("no weights")):
            out_path, ctx = tx.transform(wav)
        assert out_path == wav  # original returned, no crash


def test_upscale_failure_degrades_to_passthrough():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        _make_wav(wav, sample_rate=16000)
        eng = MagicMock()
        eng.upscale.side_effect = RuntimeError("inference blew up")
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", return_value=eng):
            out_path, ctx = tx.transform(wav)
        assert out_path == wav  # original returned, no crash


def test_context_is_preserved():
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "spoken.wav")
        _make_wav(wav, sample_rate=16000)
        tx = AudioSRTTSTransformer()
        with patch("audiosronnx.load_sr", return_value=_fake_engine()):
            _, ctx = tx.transform(wav, {"lang": "en-US"})
        assert ctx["lang"] == "en-US"
