# Architecture

This section describes the full system architecture — from raw EMG signals to
predicted keystrokes — and how every component in the codebase connects.

---

## System Overview

```mermaid
flowchart TB
    subgraph acq["📦 Data Acquisition"]
        direction TB
        MetaS3["☁️ Meta S3\n308 GB tar.gz"]
        Downloader["⬇️ EMGDownloader\nstream-filter"]
        B2["🗄️ Backblaze B2"]
        Local["💾 Local data/"]

        MetaS3 -->|"HTTPS tar.gz stream"| Downloader
        Downloader --> B2
        Downloader --> Local
        B2 -->|"rclone sync"| Local
    end

    subgraph train["🔁 Training Loop"]
        direction TB
        Dataset["🪟 WindowedEMGDataset"]
        Transforms["🔀 Transforms\nLogSpec + SpecAug"]
        Model["🧠 TDSConvEncoder"]
        CTC["📉 CTC Loss"]
        Decoder["🔤 Greedy / Beam Decoder"]
        Metric["📊 CER Metric"]

        Dataset --> Transforms
        Transforms --> Model
        Model --> CTC
        Model --> Decoder
        Decoder --> Metric
    end

    acq -->|"stream to training"| train
```

---

## Sections

| Page | Covers |
|---|---|
| [Data Layer](data-layer.md) | HDF5 session files, `EMGSessionData`, `WindowedEMGDataset`, `WindowedEMGDataModule` |
| [Transform Pipeline](transforms.md) | LogSpectrogram, SpecAugment, RandomBandRotation |
| [Model Architecture](model.md) | TDS-CNN baseline, sub-modules, hyperparameters |
| [Training Loop](training.md) | Hydra config, entry point, optimizer & LR schedule |
| [Decoding](decoding.md) | Greedy decoder, beam search, KenLM language model |
| [Metrics](metrics.md) | CER, IER, DER, SER |
| [Data Pipeline](data-pipeline.md) | EMGDownloader, B2 registry, deduplication |
| [Configuration Reference](configuration.md) | Hydra config groups, override examples |
| [Testing](testing.md) | Unit tests, integration tests, test structure |
