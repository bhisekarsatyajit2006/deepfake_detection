"""Flask web dashboard for deepfake detection.

Thin HTTP layer only — all the ML lives in the ``deepfake`` package. Run with:

    python -m webapp.app            # from the project root
"""

import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request, send_file

# Allow `python webapp/app.py` as well as `python -m webapp.app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepfake.config import settings          # noqa: E402
from deepfake.inference import DeepfakeAnalyzer  # noqa: E402
from webapp.catalog import VideoCatalog        # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

analyzer = DeepfakeAnalyzer(settings)
catalog = VideoCatalog(settings)


@app.route("/")
def index():
    return render_template("index.html", folders=catalog.folders())


@app.route("/api/folders")
def api_folders():
    return jsonify(catalog.folders())


@app.route("/api/videos/<folder>")
def api_videos(folder):
    return jsonify(catalog.videos_in(folder))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    rel_path = (request.json or {}).get("path", "")
    video_path = settings.videos_dir / rel_path
    if not video_path.exists():
        return jsonify({"error": f"File not found: {rel_path}"}), 404
    return jsonify(analyzer.analyze_video(video_path))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Frontend-facing endpoint — accepts field name 'media' (video files only).
    Returns normalised JSON compatible with the index.html result renderer.
    """
    import tempfile, os
    file = request.files.get("media") or request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided. Use field name 'media'."}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"error": f"Unsupported format '{suffix}'. Allowed: {allowed}"}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()
        print(f"[Predict] '{file.filename}' ({os.path.getsize(tmp.name)/1024/1024:.1f} MB)")
        data = analyzer.analyze_video(Path(tmp.name))
        if "error" in data:
            return jsonify(data), 422
        # Normalise field names for the frontend renderer
        probs = data.get("all_probs", [])
        threshold = data.get("threshold", 0.5)
        fake_frames = sum(1 for p in probs if p < threshold)
        real_frames = len(probs) - fake_frames
        return jsonify({
            "prediction":       data["prediction"],          # "FAKE" | "REAL"
            "verdict":          data["prediction"],
            "confidence":       data["confidence"],           # 0-100
            "avg_prob":         data["avg_prob"],
            "threshold":        threshold,
            "frames_analyzed":  data["frames_extracted"],
            "frames_extracted": data["frames_extracted"],
            "faces_detected":   data["faces_detected"],
            "deepfake_frames":  fake_frames,
            "fake_frames":      fake_frames,
            "real_frames":      real_frames,
            "all_probs":        probs,
            "prob_buckets":     data.get("prob_buckets", []),
            "video_info":       data.get("video_info", {}),
            "frame_results":    data.get("frame_results", []),
            "filename":         file.filename,
        })
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a video file and run deepfake detection on it.

    Accepts multipart/form-data with a 'video' field.
    Returns the full analysis JSON result.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use field name 'video'."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"error": f"Unsupported format '{suffix}'. Allowed: {allowed}"}), 400

    import tempfile, os
    tmp_dir = Path("D:/temp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, dir=tmp_dir, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()
        print(f"[Upload] Saved '{file.filename}' -> {tmp.name} "
              f"({os.path.getsize(tmp.name) / 1024 / 1024:.1f} MB)")
        result = analyzer.analyze_video(Path(tmp.name))
        result["filename"] = file.filename
        return jsonify(result)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/api/debug/model")
def api_debug_model():
    """GET to verify the weights loaded (probs near 0.5 => not loaded)."""
    with torch.no_grad():
        z = torch.zeros(1, 3, settings.img_size, settings.img_size, device=settings.device)
        o = torch.ones(1, 3, settings.img_size, settings.img_size, device=settings.device)
        prob_z = analyzer.model(z).item()   # model already outputs sigmoid probability
        prob_o = analyzer.model(o).item()
    return jsonify({
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "device": str(settings.device),
        "threshold": analyzer.threshold,
        "n_params": sum(p.numel() for p in analyzer.model.parameters()),
        "prob_zeros_input": round(prob_z, 4),
        "prob_ones_input": round(prob_o, 4),
        "tip": "If both probs are near 0.5, weights did not load. Check MODEL_PATH.",
    })


@app.route("/video/<path:rel_path>")
def serve_video(rel_path):
    p = settings.videos_dir / rel_path
    if not p.exists():
        return "Not found", 404
    return send_file(str(p), mimetype="video/mp4")


def main():
    print(f"[Config] VIDEOS_DIR = {settings.videos_dir}")
    print(f"[Config] MODEL_PATH = {settings.model_path}")
    print(f"[Config] CSV_PATH   = {settings.csv_path}")
    app.run(debug=True, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
