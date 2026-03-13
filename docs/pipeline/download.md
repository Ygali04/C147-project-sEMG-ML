# Download Script

The download script (`scripts/download_data.py`) streams HDF5 session files
from Meta's public S3 tar.gz archive, saves them locally, and uploads them to
Backblaze B2.

## Usage

```bash
# Download baseline profile (user 89335547, 18 sessions)
uv run python scripts/download_data.py --baseline

# Download 10 random user profiles
uv run python scripts/download_data.py --test

# Download ALL profiles (~200 GB, streams entire 308 GB tar.gz)
uv run python scripts/download_data.py --all

# Preview what would be downloaded without uploading
uv run python scripts/download_data.py --test --dry-run

# Custom number of test users with a specific seed
uv run python scripts/download_data.py --test --n-test-users 5 --seed 42
```

Or use the Makefile shortcuts:

```bash
make download-baseline
make download-test
make download-all
```

## Prerequisites

1. **Backblaze B2 credentials** — set `B2_KEY_ID` and `B2_APPLICATION_KEY` in
   your `.env` file:

    ```bash
    cp .env.example .env
    # Edit .env with your credentials
    ```

2. **Dependencies** — install with:

    ```bash
    uv sync --group dev
    ```

## How It Works

### 1. Session Resolution

Based on the `--mode` flag, the script determines which `(user_id, session_id)`
pairs to download:

| Mode | Strategy |
|---|---|
| `--baseline` | Hardcoded list of 18 sessions for user `89335547` |
| `--test` | Samples `n_test_users` random users from `metadata.csv` |
| `--all` | Uses all entries in `metadata.csv` |

The `metadata.csv` is downloaded once from Meta's S3 and cached locally in
`data/metadata.csv`.

### 2. Registry Check

A JSON manifest (`emg2qwerty_registry.json`) stored in the B2 bucket tracks
which files have already been uploaded. The downloader also checks whether each
file exists locally in `data/`. Files present in **either** location are skipped.

### 3. Streaming Tar.gz Extraction

The Meta dataset is distributed as a single gzip-compressed tar archive:

```
https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
```

The downloader streams this file over HTTPS using `requests` and processes it
with `tarfile.open(fileobj=response.raw, mode='r|gz')`. As each tar member is
encountered, it checks if the member name matches a pending session. If so, the
file content is extracted into memory, saved locally, and uploaded to B2.

!!! info "Streaming semantics"
    The `r|gz` mode processes the archive sequentially — members cannot be
    seeked to. This means the entire archive is streamed even if only a few
    files are needed. However, non-matching members are skipped without
    reading their content, keeping memory usage minimal.

### 4. Upload to B2

Each extracted HDF5 file is uploaded to B2 using `boto3.put_object`:

```
B2 bucket: C147-project
Key format: emg2qwerty/<user_id>/<session_id>.hdf5
```

### 5. Registry Update

After each successful upload, a `FileRecord` is created with:

- `source_key` — path inside the tar archive
- `b2_key` — object key in B2
- `size_bytes` — file size
- `etag` — MD5 hash of the content
- `uploaded_at` — ISO 8601 timestamp

The registry is saved back to B2 after all processing completes.

## File Registry

The registry is a JSON file stored at the root of the B2 bucket:

```json
{
  "emg2qwerty/89335547/2021-06-03-session.hdf5": {
    "source_key": "emg2qwerty-data-2021-08/89335547/2021-06-03-session.hdf5",
    "b2_key": "emg2qwerty/89335547/2021-06-03-session.hdf5",
    "size_bytes": 524288000,
    "etag": "\"abc123def456\"",
    "uploaded_at": "2026-03-11T12:00:00+00:00"
  }
}
```

This ensures that re-running the download script is **idempotent** — only new
files are transferred.

## CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--baseline` | flag | ✓ | Download baseline user (89335547) |
| `--test` | flag | | Download random test users |
| `--all` | flag | | Download all users |
| `--dry-run` | flag | off | Preview without downloading |
| `--n-test-users` | int | 10 | Users to sample in `--test` mode |
| `--seed` | int | 1501 | Random seed for sampling |

## rclone Alternative

For manual access or mounting the B2 bucket as a local filesystem:

```bash
# One-time setup
make rclone-config

# List files
rclone ls b2-c147:C147-project/emg2qwerty/

# Sync to local
rclone sync b2-c147:C147-project/emg2qwerty/ data/emg2qwerty/ --progress

# Mount (read-only)
rclone mount b2-c147:C147-project/emg2qwerty/ data/emg2qwerty/ --read-only
```
