# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Any, Dict, Optional, Tuple

from ovos_utils.log import LOG
from ovos_plugin_manager.templates.transformers import TTSTransformer

from .version import __version__

# audiosronnx produces 48 kHz output regardless of the engine.
OUTPUT_SAMPLE_RATE = 48000


class AudioSRTTSTransformer(TTSTransformer):
    """Audio super-resolution for any TTS plugin.

    Runs after the TTS stage and before playback. The synthesized ``.wav`` is
    passed through an ``audiosronnx`` super-resolution engine and re-written at
    48 kHz, recovering high-frequency detail that low-sample-rate voices discard.
    It operates on the waveform, so it is engine-agnostic — it upscales the output
    of whichever TTS plugin is active.

    Configured under ``tts_transformers`` in ``mycroft.conf``::

        "tts_transformers": {
          "ovos-tts-transformer-audiosr": {"engine": "novasr"}
        }

    The engine is loaded once on first use and cached. If ``audiosronnx`` (or its
    weights) is unavailable, or an upscale fails, the original audio is returned
    unchanged so synthesis never breaks.
    """

    def __init__(self, name: str = "ovos-tts-transformer-audiosr",
                 priority: int = 50, config: Optional[dict] = None):
        super().__init__(name, priority, config)
        # novasr is the default: natural on already-wideband TTS output and
        # lightweight. lavasr/hifiganbwe/apbwe are also available.
        self.engine_name = self.config.get("engine", "novasr")
        self._sr_engine = None
        self._sr_loaded = False

    def _load_engine(self):
        """Lazily load and cache the audiosronnx super-resolution engine.

        Returns the engine, or ``None`` when audiosronnx is missing or the engine
        fails to load (a warning is logged and the transform degrades to a no-op).
        """
        if self._sr_loaded:
            return self._sr_engine
        self._sr_loaded = True
        try:
            from audiosronnx import load_sr
        except ImportError:
            LOG.warning(
                "ovos-tts-transformer-audiosr is enabled but audiosronnx is not "
                "installed; TTS audio will be passed through unchanged. "
                "Install it with: pip install ovos-tts-transformer-audiosr"
            )
            return None
        try:
            self._sr_engine = load_sr(engine=self.engine_name)
            LOG.info(f"Audio super-resolution enabled (engine={self.engine_name}); "
                     "TTS output will be upscaled to 48 kHz")
        except Exception as exc:
            LOG.warning(f"Failed to load super-resolution engine "
                        f"'{self.engine_name}': {exc}. TTS audio will be passed "
                        "through unchanged.")
            self._sr_engine = None
        return self._sr_engine

    def transform(self, wav_file: str,
                  context: Optional[dict] = None) -> Tuple[str, Dict[str, Any]]:
        """Upscale the TTS ``wav_file`` to 48 kHz and return the new file path.

        The returned wav carries its sample rate in its own header, so downstream
        playback picks up 48 kHz without any out-of-band signalling. Audio already
        at (or above) 48 kHz is returned untouched, as is the original file on any
        failure.

        :param wav_file: path to the wav generated in the TTS stage.
        :returns: ``(path_to_wav, context)`` — the upscaled wav, or the original.
        """
        context = context or {}
        engine = self._load_engine()
        if engine is None:
            return wav_file, context
        try:
            from audiosronnx.audio import read_wav, write_wav
            audio, in_sr = read_wav(wav_file)
            if in_sr >= OUTPUT_SAMPLE_RATE:
                # already hi-res — nothing to gain
                return wav_file, context
            out, out_sr = engine.upscale(audio, in_sr)
            if wav_file.lower().endswith(".wav"):
                outpath = wav_file[:-4] + "_sr.wav"
            else:
                outpath = wav_file + "_sr.wav"
            write_wav(outpath, out, out_sr)
            LOG.debug(f"Upscaled TTS audio {in_sr} Hz -> {out_sr} Hz: {outpath}")
            return outpath, context
        except Exception as exc:
            LOG.warning(f"Super-resolution failed ({exc}); playing native "
                        "TTS audio instead")
            return wav_file, context


if __name__ == "__main__":
    tx = AudioSRTTSTransformer(config={"engine": "novasr"})
    out, _ = tx.transform("test.wav")
    print(out)
