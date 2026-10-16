
import { useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  CircleAlert,
  Clock3,
  ImagePlus,
  Leaf,
  LoaderCircle,
  ScanSearch,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function App() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  function chooseFile(selectedFile) {
    if (!selectedFile) return;

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
      "image/bmp",
    ];

    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please choose a JPG, PNG, WEBP, or BMP image.");
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError("Your image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function analyseImage() {
    if (!file) {
      setError("Upload a leaf image before starting detection.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The image could not be analysed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot reach the backend. Make sure FastAPI is running at http://127.0.0.1:8000."
          : err.message || "Something went wrong during detection."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home">
          <span className="brand-mark">
            <Leaf size={23} />
          </span>
          <span>
            <strong>Potato Vision</strong>
            <small>COMPUTER VISION LAB</small>
          </span>
        </a>

        <div className="side-label">WORKSPACE</div>
        <a className="nav-item active" href="#analysis">
          <ScanSearch size={18} />
          Image analysis
        </a>

        <div className="side-label">SYSTEM</div>
        <div className="system-status">
          <span className="status-dot" />
          <span>
            <strong>Inference API</strong>
            <small>Local development</small>
          </span>
        </div>

        <div className="sidebar-bottom">
          <ShieldCheck size={18} />
          <span>
            <strong>Research prototype</strong>
            <small>Predictions require verification</small>
          </span>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>Image analysis</strong>
          </div>
          <div className="local-badge">
            <span className="status-dot" />
            Running locally
          </div>
        </header>

        <section className="page-heading" id="analysis">
          <div>
            <div className="eyebrow">
              <Activity size={15} /> POTATO LEAF INTELLIGENCE
            </div>
            <h1>See beyond the surface.</h1>
            <p>
              Upload a potato-leaf image to identify regions flagged by
              your late-blight detection model.
            </p>
          </div>
          <div className="heading-icon">
            <Leaf size={34} />
          </div>
        </section>

        <section className="workflow">
          <div className="workflow-step current">
            <span>01</span> Upload image
          </div>
          <ArrowRight size={16} />
          <div className={`workflow-step ${result ? "complete" : ""}`}>
            <span>02</span> Run detection
          </div>
          <ArrowRight size={16} />
          <div className={`workflow-step ${result ? "complete" : ""}`}>
            <span>03</span> Review results
          </div>
        </section>

        <div className="workspace-grid">
          <section className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <div className="panel-kicker">INPUT</div>
                <h2>Leaf image</h2>
              </div>
              <ImagePlus size={21} />
            </div>

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    fileInputRef.current?.click();
                  }
                }}
              >
                <div className="upload-icon">
                  <Upload size={24} />
                </div>
                <h3>Drop your image here</h3>
                <p>or browse files on your device</p>
                <span className="browse-button">Choose image</span>
                <small>JPG, PNG, WEBP · Up to 10 MB</small>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Uploaded potato leaf" />
                <button
                  className="remove-image"
                  onClick={clearImage}
                  aria-label="Remove selected image"
                  type="button"
                >
                  <X size={18} />
                </button>
                <div className="image-caption">
                  <div>
                    <strong>{file.name}</strong>
                    <small>
                      {(file.size / 1024).toFixed(1)} KB · Original image
                    </small>
                  </div>
                  <CheckCircle2 size={20} />
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,image/*"
              hidden
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {error && (
              <div className="error-message">
                <CircleAlert size={17} />
                <span>{error}</span>
              </div>
            )}

            <button
              className="analyse-button"
              onClick={analyseImage}
              disabled={!file || loading}
              type="button"
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={19} />
                  Analysing image...
                </>
              ) : (
                <>
                  <ScanSearch size={19} />
                  Detect disease regions
                  <ArrowRight size={18} />
                </>
              )}
            </button>

            <div className="privacy-note">
              <ShieldCheck size={16} />
              Image processed by your local inference API.
            </div>
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <div>
                <div className="panel-kicker">OUTPUT</div>
                <h2>Detection results</h2>
              </div>
              <span className={`result-badge ${result ? "ready" : ""}`}>
                {result ? "Analysis complete" : "Awaiting image"}
              </span>
            </div>

            {!result && !loading && (
              <div className="empty-results">
                <div className="empty-illustration">
                  <ScanSearch size={38} />
                  <span />
                </div>
                <h3>Your results will appear here</h3>
                <p>
                  Upload an image and run detection to see the model's
                  predictions and annotated image.
                </p>
                <div className="empty-tip">
                  <Clock3 size={16} />
                  Results appear after inference completes
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <LoaderCircle className="spin" size={35} />
                <h3>Examining your image</h3>
                <p>Running the trained YOLO detector on your image.</p>
              </div>
            )}

            {result && (
              <div className="results-content">
                <div className="summary-card">
                  <div>
                    <small>REGIONS DETECTED</small>
                    <div className="big-number">
                      {result.detection_count}
                    </div>
                  </div>
                  <div className="summary-icon">
                    <ScanSearch size={24} />
                  </div>
                </div>

                <div className="annotated-image">
                  <img
                    src={result.annotated_image}
                    alt="Leaf image with YOLO detection boxes"
                  />
                </div>

                <div className="image-dimensions">
                  <span>{result.filename}</span>
                  <span>
                    {result.image_width} × {result.image_height} px
                  </span>
                </div>

                <h3 className="subheading">Detected regions</h3>

                {result.detection_count === 0 ? (
                  <div className="no-detections">
                    No regions were detected at the current confidence
                    threshold. This does not mean the leaf is healthy.
                  </div>
                ) : (
                  <div className="detection-list">
                    {result.detections.map((detection, index) => (
                      <div
                        className="detection-row"
                        key={`${index}-${detection.class_id}`}
                      >
                        <div className="detection-symbol">
                          <ScanSearch size={18} />
                        </div>
                        <div className="detection-info">
                          <strong>{detection.class_name}</strong>
                          <small>
                            Region {index + 1} · Bounding box
                          </small>
                        </div>
                        <div className="confidence">
                          {(detection.confidence * 100).toFixed(1)}%
                          <small>confidence</small>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                <div className="disclaimer">
                  <CircleAlert size={17} />
                  <p>
                    {result.notice} This is an experimental model output,
                    not a confirmed diagnosis.
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>

        <footer className="footer">
          <span>Potato Vision · Research prototype</span>
          <span>YOLO · Local inference</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
