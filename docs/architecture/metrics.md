# Metrics

The primary evaluation metric is **Character Error Rate (CER)**:

\[
\text{CER} = \frac{\text{edit\_distance}(\text{prediction}, \text{reference})}{\text{len}(\text{reference})}
\]

The `CharacterErrorRates` module (`metrics.py`) also reports decomposed error types:

| Metric | Meaning |
|---|---|
| **CER** | Overall character error rate |
| **IER** | Insertion error rate |
| **DER** | Deletion error rate |
| **SER** | Substitution error rate |

These metrics satisfy:

\[
	\text{CER} = \text{IER} + \text{DER} + \text{SER}
\]

when each term is normalized by the reference sequence length.

## How to interpret them

- High **IER** means the decoder is emitting extra characters.
- High **DER** means the model is missing characters that should be present.
- High **SER** means alignments are mostly correct, but the predicted character is often wrong.

In this project, checkpoints are selected with `val/CER`, while DER, IER, and
SER are most useful for diagnosing what kind of mistakes remain after overall
CER has improved.

## Example: current best documented recurrent run

For the `cnn_bilstm_ctc` single-user run on user 89335547:

| Split | CER (%) | DER (%) | IER (%) | SER (%) | Loss |
|---|---|---|---|---|---|
| Validation | 13.76 | 1.77 | 3.15 | 8.84 | 0.544 |
| Test | 14.89 | 1.36 | 2.64 | 10.89 | 0.556 |

The main remaining error source is substitution, not insertion or deletion.
