## Global Constraints

- Language: **Rust**, one binary crate named `ferret`, edition 2024. Modules, not a workspace of crates.
- Ferret never maps model weights. If the embedder is down, index and search fail pointing at `embedder.endpoint`.
- v1 file kinds: only `text|image|pdf`. Chunk kinds: only `text|image|pdf_page`.
- No in-tree GUI, watcher, daemon/socket, audio, video, OCR, Office readers, Ollama backend, or vendored GGUFs/llama.cpp/LiteRT/ImageMagick (default package).
- Do not use FP16 anywhere ferret controls. Store only finite `f32`. L2-normalize every vector.
- `model_id` in index meta **is** `embedder.model`. Opening an index whose `model_id`, `dimensions`, or `prefix_profile` differ is exit 1.
- Filters apply in SQL before ranking. vec0 metadata: `file_kind`/`ext`/`mtime`/`collection`. `--under`/`--name` JOIN `files` with vec `k = min(candidates*8, 400)`.
- JSON `score` is fused RRF. JSON `scores.vec` is cosine `1 - (distance^2)/2` clamped `[0,1]`. Do not print fused RRF as a percent.
- `--name` SQL `GLOB` is **case-sensitive**, including on APFS.
- Never mutate corpus files. Never write PDF rasters into the corpus tree. Cache under `{cache_dir}/pages/`.
- HTTP: 3 attempts, linear backoff 200ms/400ms/800ms on 429/503/connection reset. Other 4xx are not retried.
- Exit 0: success (zero hits, some per-file index errors, `rm` of a missing path). Exit 1: fatal (config, DB, embedder, schema/model mismatch, `--config` missing file). Exit 2: usage / unknown flag / `init` without `--force` onto an existing file / invalid date / `--under` DIR does not exist.
- SQLite ≥ 3.38 via rusqlite `bundled`. sqlite-vec **0.1.x** registered in-process; no runtime `.so` load.
- `FERRET_EMBEDDER=fake` wins over `embedder.backend`; fake sets `images_supported = true`.
- Path policy: `dirs` crate defaults; if any `XDG_*` is set, that variable wins on **every** OS. macOS `config_dir` default is `~/Library/Preferences/ferret` (not Application Support).
- CI does not download models or call a GPU.
- The repo may not be a git repository yet. Task 1 runs `git init`. Commit steps assume that. If the operator has forbidden commits, skip only the commit step and continue.

---
