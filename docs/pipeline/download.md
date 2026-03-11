# Download Script

The download script (`scripts/download_data.py`) streams HDF5 session files from Meta's public S3 bucket directly into Backblaze B2 without using local disk space.

## Usage

```bash
# Download baseline profile (user 89335547, 18 sessions)
uv run python scripts/download_data.py --baseline

# Download 10 random user profiles
uv run python scripts/download_data.py --test

# Download ALL profiles (~200 GB)
uv run python scripts/download_data.py --all

# Preview what would be downloaded without uploading
uv run python scripts/download_data.py --test --dry-run

# Custom number of test users
uv run python scripts/download_data.py --test --n-test-users 5 --seed 42
```

Or use the Makefile shortcuts:

```bash
make download-baseline
make download-test
make download-all
```

## Prerequisites

1. **Backblaze B2 credentials** — set `B2_KEY_ID` and `B2_APPLICATION_KEY` in your `.env` file:

    ```bash
    cp .env.example .env
    # Edit .env with your credentials
    ```

2. **Dependencies** — install with:

    ```bash
    uv sync --group dev
    ```

## How It Works

1. **Session resolution** — Based on the `--mode` flag, the script determines which user/session pairs to download:
    - `--baseline`: reads `config/user/single_user.yaml`
    - `--test`: samples `n_test_users` random users from `metadata.csv`
    - `--all`: uses all entries in `metadata.csv`

2. **Registry check** — A JSON manifest (`emg2qwerty_registry.json`) stored in the B2 bucket tracks which files have already been uploaded. Files present in the registry are skipped.

3. **Streaming transfer** — Each HDF5 file is streamed directly from Meta S3 to B2 using boto3 (`get_object` → `upload_fileobj`). No local disk is needed.

4. **Registry update** — After each successful upload, the registry is updated and saved back to B2.

## File Registry

The registry is a JSON file stored at the root of the B2 bucket:

```json
{
  "emg2qwerty/89335547/2021-06-03-session.hdf5": {
    "source_key": "emg2qwerty/89335547/2021-06-03-session.hdf5",
    "b2_key": "emg2qwerty/89335547/2021-06-03-session.hdf5",
    "size_bytes": 524288000,
    "etag": "\"abc123\"",
    "uploaded_at": "2026-03-11T12:00:00+00:00"
  }
}
```

This ensures that re-running the download script is idempotent — only new files are transferred.

## rclone Alternative

For manual access or mounting the B2 bucket as a local filesystem:

```bash
# One-time setup
make rclone-setup

# List files
rclone ls b2-c147:C147-project/emg2qwerty/

# Sync to local
rclone sync b2-c147:C147-project/emg2qwerty/ data/emg2qwerty/ --progress

# Mount (read-only)
rclone mount b2-c147:C147-project/emg2qwerty/ data/emg2qwerty/ --read-only
```
