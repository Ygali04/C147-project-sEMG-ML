# Decoding

---

## `CTCGreedyDecoder` (`decoder.py`)

The default decoder. Performs argmax at each timestep, then collapses
consecutive duplicates and removes blanks. No external dependencies required.

```bash
uv run python -m emg2qwerty.train decoder=ctc_greedy ...
```

---

## `CTCBeamDecoder` (`decoder.py`)

Beam search with optional n-gram language model (KenLM) rescoring.

| Parameter | Default | Description |
|---|---|---|
| `beam_size` | 50 | Number of beams |
| `max_labels_per_timestep` | 10 | Labels expanded per step |
| `lm_path` | `models/lm/wikitext-103-6gram-charlm.bin` | KenLM binary |
| `lm_weight` | 2.0 | LM score weight |
| `insertion_bonus` | 2.0 | Bonus for inserting characters |
| `delete_key` | `Key.backspace` | Character mapped to deletion |

```bash
uv run python -m emg2qwerty.train decoder=ctc_beam ...
```

!!! note "KenLM required"
    The beam-search decoder requires [KenLM](https://github.com/kpu/kenlm) to
    be installed. See [Setup → KenLM](../getting-started/setup.md#kenlm-beam-search-decoder)
    for installation instructions.

---

## Building the Character Language Model

The 6-gram character LM is built from WikiText-103:

```bash
# Build kenlm C++ tools first:
# https://github.com/kpu/kenlm#compiling

# Then build the 6-gram char LM:
./scripts/lm/build_char_lm.sh 6
```

This produces:

- `models/lm/wikitext-103-6gram-charlm.arpa` — human-readable ARPA format
- `models/lm/wikitext-103-6gram-charlm.bin` — fast binary format (used at inference)
