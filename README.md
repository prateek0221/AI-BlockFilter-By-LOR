# AI-BlockFilter-By-LOR

**Elevator pitch**

AI-BlockFilter-By-LOR is an opinionated, production-minded repo that turns continuous NVR/video streams into *meaningful blocks* by detecting people and filtering video segments based on user-defined Lines/Regions-of-Interest (LOR). It’s built for operators who want fewer false alarms and faster review: automatically extract, compress, label, and store only the segments where people cross or appear within the LOR boundaries.

---

## Highlights & Creative Description

Imagine a security operator who only wants to see the moments that matter. AI-BlockFilter-By-LOR uses a modern person-detector (YOLOv8 / OpenVINO / TensorRT friendly), a lightweight tracker, and an LOR-based spatial rule engine that lets you draw a line or rectangular region on camera view and treat that as the filter. When a person crosses or stays inside the region, the system produces a video block (MP4), a thumbnail, and a JSON event metadata record — ready to index and review.

It is:

* **Accurate** — uses SOTA person detector with NMS+tracker to reduce duplicate clips.
* **Configurable** — per-camera LOR definitions, min-duration, entry/exit policies, confidence thresholds.
* **Efficient** — only stores filtered blocks (configurable compression + length limits).
* **Auditable** — every block includes JSON metadata for downstream search and classifiers.

---

## Repository structure

```
AI-BlockFilter-By-LOR/
├── README.md                # This file (human-friendly guide)
├── LICENSE
├── .gitignore
├── config/
│   └── cameras.json         # Example camera + LOR config (see below)
├── src/
│   ├── main.py              # entrypoint (example: runs on folders or RTSP)
│   ├── detector.py          # wrapper for person detector (YOLOv8 / ONNX)
│   ├── tracker.py           # lightweight tracker wrapper (sort/strongsort)
│   ├── lor_engine.py        # LOR logic: crossing, dwell, direction
│   ├── extractor.py         # extracts + compresses clips using ffmpeg
│   └── utils.py
├── examples/
│   ├── sample_videos/       # small sample video(s) used for testing
│   └── demo.sh              # small demo script
├── docs/
│   └── architecture.md
└── docker/
    └── Dockerfile
```

---

## Quick install (local, dev)

1. Clone:

```bash
git clone https://github.com/your-org/AI-BlockFilter-By-LOR.git
cd AI-BlockFilter-By-LOR
```

2. Create virtual environment and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` should include `opencv-python-headless`, `torch` (or ONNX runtime), `ultralytics` (if YOLOv8), `ffmpeg-python`, and minimal tracker dependencies.

3. Prepare camera config: `config/cameras.json` (example below).

4. Run demo (process a folder of videos):

```bash
python src/main.py --config config/cameras.json --input examples/sample_videos --output out/blocks
```

Or stream from RTSP:

```bash
python src/main.py --config config/cameras.json --rtsp rtsp://user:pass@cam-ip:554/stream --output out/blocks
```

---


> `lor.type` supports `line` (2 points), `rect` (two corners), or `polygon` (N points). `trigger_on` can be `crossing`, `presence`, or `dwell`.

---

## Example lor.json (JSON metadata)

```json

{
  "8": [
    [
      301,
      280,
      331,
      470
    ]
  ],
  "39": [
    [
      237,
      225,
      266,
      483
    ],
    [
      404,
      380,
      396,
      437
    ],
    [
      398,
      445,
      249,
      444
    ]
  ]
}
```
where "8", "39" are cams
---

## Architecture notes

* **Detector module**: abstracted so you can plug ONNX, TorchScript, or PyTorch weights. Accepts frames and returns person bboxes + scores.
* **Tracker**: optional; recommended for smoothing and de-duplicating repeated detections. Use SORT/StrongSORT.
* **LOR Engine**: geometry utilities that check if a bbox center crosses a line, enters a rect, or stays inside a polygon for `dwell_secs`.
* **Extractor**: when an event is triggered, the extractor buffers preceding N seconds and subsequent N seconds, stitches frames, and creates compressed block using ffmpeg.
* **Storage**: filesystem-first but pluggable to S3, Minio, or NFS.

---

## How to tune

* Raise `confidence_threshold` to reduce false positives
* Use `min_duration_secs` to ignore rapid noises
* Enable `tracker` to merge multiple detections of same person
* Use `dwell_secs` inside rect LOR to require a person to pause before triggering

---

## Demo & testing tips

* Use `examples/sample_videos/` to run quick offline tests
* Use `demo.sh` to run the pipeline in single-process mode
* For scale, run each camera process as its own container / process and publish JSON events to Kafka or webhook

---

## Contributing

Please open issues for feature requests and pull requests against `develop`. Follow the `CODE_OF_CONDUCT.md` and tests must be added for significant features.

---

## License

MIT

---

## Next steps I can do for you (pick any and I will create them right away):

1. Generate the complete `src/` skeleton (Python) with TODOs and docstrings.
2. Create Dockerfile + CI workflow (GitHub Actions) to build and smoke-test.
3. Produce a sample small test video (clip) and show the pipeline run locally (video not included in repo by default).

---

*Created for you — let me know which follow-up you want and I will add the files directly into the repo.*
