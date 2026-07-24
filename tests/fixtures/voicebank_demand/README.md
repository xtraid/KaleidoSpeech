# VoiceBank-DEMAND test subset

Four paired samples from the public VoiceBank-DEMAND test set are included:

| ID | Demo noise tranche | Transcript |
|---|---:|---|
| `p232_036` | 1 (most affected) | But it may take some time to confirm the findings. |
| `p232_080` | 2 | I've got the shirt. |
| `p232_145` | 3 | We want to see the maximum of change. |
| `p232_075` | 4 | The price cuts are really exciting. |

Directory layout:

- `source/noisy`: original noisy WAV from the public demo;
- `source/clean`: paired clean reference WAV;
- `redis`: normalized 16 kHz source material used to generate signed int16
  little-endian Redis frames during the tests;
- `reference`: clean audio converted in the same way, used only as an oracle;
- `transcripts`: official test-set transcripts.

The test suite quantizes `redis/<id>.f32` to signed int16, splits it into 40 ms
frames and builds records with the exact `sequence`, `captured_at_ns`,
`sample_rate`, `frame_ms` and `pcm_s16le` fields used by the application.
Only the aggregated `pcm_s16le` word window reaches the cleaner. Neither the
clean reference nor the transcript is visible to it.

Source: Cassia Valentini-Botinhao, *Noisy speech database for training speech
enhancement algorithms and TTS models*, University of Edinburgh, 2017,
<https://doi.org/10.7488/ds/2117>.

Audio samples were retrieved from the public VoiceBank-DEMAND demonstration at
<https://miyazaki-lab.github.io/icassp2020_demo/> and technically converted with:

```bash
ffmpeg -nostdin -v error -i input.wav -ac 1 -ar 16000 -f f32le output.f32
```

The dataset is distributed under the Creative Commons Attribution 4.0
International licence. The files here are an attributed, technically converted
subset.
