# OVOS AudioSR TTS Transformer

Engine-agnostic audio super-resolution for OpenVoiceOS speech synthesis. It
upscales the audio produced by **any** TTS plugin to 48 kHz just before playback,
recovering high-frequency detail that low-sample-rate voices discard. The result
is brighter, less muffled speech without retraining or replacing your existing
voice.

This is a [TTS transformer](https://openvoiceos.github.io/ovos-technical-manual/tts_transformers/)
plugin (`opm.transformer.tts`). It runs after the TTS stage and before playback.
It operates on the generated waveform, not on text, so it works with any TTS engine.

It wraps [`audiosronnx`](https://github.com/TigreGotico/audiosronnx), a pure-ONNX
super-resolution library (no Torch at runtime).

## How it works

- The generated `.wav` is read and, unless it is already 48 kHz, passed to the
  configured `audiosronnx` engine.
- The engine reconstructs a 48 kHz waveform from the low-resolution input.
- The upscaled audio is written alongside the original (`<name>_sr.wav`) and
  handed back for playback. Its 48 kHz rate travels in the wav header.
- If audio is already 48 kHz, the transform is skipped and the file is returned
  untouched.
- If `audiosronnx` (or its weights) is unavailable, or an upscale fails, the
  original audio is returned unchanged. Synthesis never breaks.

Model weights are fetched from the Hugging Face Hub on first use and cached
locally. No manual download step is required.

## Installation

```bash
pip install ovos-tts-transformer-audiosr
```

## Configuration

Enable the transformer in `mycroft.conf` under the `tts_transformers` section,
keyed by the plugin name:

```json
"tts_transformers": {
  "ovos-tts-transformer-audiosr": {"engine": "novasr"}
}
```

Multiple TTS transformers can be chained. Execution order follows each plugin's
`priority` (this plugin defaults to `50`).

### Engines

Select the super-resolution engine with the `engine` config key:

| engine        | notes                                                                 |
|---------------|-----------------------------------------------------------------------|
| `novasr`      | **Default.** Natural on TTS output and lightweight.                    |
| `lavasr`      | Higher-detail bandwidth extension. Can add audible high-frequency noise on already-wideband voices. |
| `hifiganbwe`  | HiFi-GAN bandwidth extension.                                         |
| `apbwe`       | Amplitude-and-phase bandwidth extension.                             |

`novasr` is the default because it stays natural on synthesized speech, which is
often already fairly wideband. `lavasr` recovers more detail but can introduce
high-frequency noise on such input. All engines output 48 kHz. Model download
sizes are modest, from a few to a few tens of MB. See the
[`audiosronnx`](https://github.com/TigreGotico/audiosronnx) project for the
per-engine model details.

## Requirements

- [`audiosronnx`](https://github.com/TigreGotico/audiosronnx) (pure `onnxruntime`)
- `ovos-plugin-manager`

## Related

- [`ovos-tts-transformer-sox-plugin`](https://github.com/OpenVoiceOS/ovos-tts-transformer-sox-plugin):
  general-purpose audio effects (pitch, reverb, EQ, and more) for TTS output.

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).
