# Testing

---

## Run All Unit Tests

```bash
make test
# or equivalently:
uv run pytest -p no:cacheprovider -m "not slow and not integration" -n auto
```

---

## Run Integration Tests (requires B2 credentials + network)

```bash
uv run pytest -m integration -v -s
```

---

## Test Structure

| File | Covers |
|---|---|
| `test_charset.py` | Key ↔ label ↔ unicode mapping roundtrips |
| `test_data.py` | Label data creation from keystroke strings |
| `test_decoder.py` | Greedy + beam decoders, kenlm LM scoring |
| `test_pipeline_config.py` | Pydantic config validation (B2, download, training) |
| `test_pipeline_registry.py` | FileRegistry CRUD, save/load roundtrip (moto-mocked S3) |
| `test_pipeline_downloader.py` | EMGDownloader session resolution, dry-run, dedup |
| `test_train_batched.py` | BatchTrainer profile resolution, Hydra overrides |
| `test_integration_baseline.py` | End-to-end: B2 connectivity, tar.gz stream, HDF5 validation |
