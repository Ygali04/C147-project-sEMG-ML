# Metrics

The primary evaluation metric is **Character Error Rate (CER)**:

\[
\text{CER} = \frac{\text{edit_distance}(\text{prediction}, \text{reference})}{\text{len}(\text{reference})}
\]

The `CharacterErrorRates` module (`metrics.py`) also reports decomposed error types:

| Metric | Meaning |
|---|---|
| **CER** | Overall character error rate |
| **IER** | Insertion error rate |
| **DER** | Deletion error rate |
| **SER** | Substitution error rate |
