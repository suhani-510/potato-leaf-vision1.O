import { useEffect, useRef, useState } from "react";
import {
  CircleAlert,
  Download,
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
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

const VIEWS = [
  { id: "original", label: "Original", legend: [] },
  {
    id: "detection",
    label: "Detection",
    legend: [{ kind: "box", text: "YOLO detection boxes" }],
  },
  {
    id: "segmentation",
    label: "Segmentation",
    legend: [{ kind: "fill", text: "OpenCV candidate regions" }],
  },
  {
    id: "combined",
    label: "Combined",
    legend: [
      { kind: "fill", text: "OpenCV candidate regions" },
      { kind: "box", text: "YOLO detection boxes" },
    ],
  },
];

const IMAGE_ALT = {
  original: "Uploaded potato leaf",
  detection: "Potato leaf with YOLO detection boxes",
  segmentation: "Potato leaf with heuristic candidate-region overlay",
  combined: "Potato leaf with candidate-region overlay and YOLO boxes",
};

function getImageSrc(imageData) {
  if (!imageData || typeof imageData !== "string") {
    return "";
  }

  const trimmed = imageData.trim();

  if (trimmed.startsWith("data:image/")) {
    return trimmed;
  }

  return `data:image/png;base64,${trimmed.replace(/\s/g, "")}`;
}

function formatSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function App() {
  const fileInputRef = useRef(null);
  const previewRef = useRef("");

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [activeView, setActiveView] = useState("combined");
  const [imageError, setImageError] = useState("");
  const [apiStatus, setApiStatus] = useState("idle");

  // Release the object URL when the component unmounts.
  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  function chooseFile(selectedFile) {
    if (!selectedFile) return;

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError("Choose a JPG, PNG, WEBP or BMP image.");
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError("That image is larger than 10 MB. Choose a smaller one.");
      return;
    }

    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
    }

    const url = URL.createObjectURL(selectedFile);
    previewRef.current = url;

    setFile(selectedFile);
    setPreview(url);
    setResult(null);
    setError("");
    setImageError("");
    setActiveView("combined");
  }

  function clearImage() {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    }

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setImageError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function analyseImage() {
    if (!file) {
      setError("Upload a leaf image before running the analysis.");
      return;
    }

    setLoading(true);
    setError("");
    setImageError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      setApiStatus("online");

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "The image could not be analysed."
        );
      }

      if (!data.annotated_image && !data.segmentation_image && !data.combined_image) {
        throw new Error(
          "The API returned no displayable images. Check the /predict response."
        );
      }

      setResult(data);
      setActiveView("combined");
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setApiStatus("offline");
        setError(
          "Cannot reach the backend. Start FastAPI at http://127.0.0.1:8000 and try again."
        );
      } else {
        setError(err.message || "Something went wrong during detection.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  function selectView(id) {
    setActiveView(id);
    setImageError("");
  }

  const view = VIEWS.find((item) => item.id === activeView) ?? VIEWS[3];

  const imageSrc = !result
    ? preview
    : activeView === "original"
      ? preview
      : getImageSrc(
          {
            detection: result.annotated_image,
            segmentation: result.segmentation_image,
            combined: result.combined_image,
          }[activeView]
        );

  const detectionCount =
    result?.detection_count ?? result?.detections?.length ?? 0;

  const width = result?.width ?? result?.image_width;
  const height = result?.height ?? result?.image_height;
  const stats = result?.segmentation_stats;

  const apiLabel = {
    idle: "Local API, not checked yet",
    online: "Local API connected",
    offline: "Local API unreachable",
  }[apiStatus];

  const downloadName = `${(result?.filename ?? file?.name ?? "leaf").replace(
    /\.[^.]+$/,
    ""
  )}-${activeView}.png`;

  return (
    <div className="page">
      <header className="header">
        <a className="brand" href="#top">
          <span className="brand-mark">
            <Leaf size={20} strokeWidth={2.2} />
          </span>
          <span className="brand-name">Potato Vision</span>
        </a>

        <div className={`api-pill api-${apiStatus}`} role="status">
          <span className="api-dot" />
          {apiLabel}
        </div>
      </header>

      <main className="content" id="top">
        <section className="intro">
          <h1>Check a potato leaf for late blight</h1>
          <p>
            Upload a photo and compare what the YOLO model detects with the
            regions found by image processing.
          </p>
        </section>

        <div className="layout">
          {/* ---------- Left: input ---------- */}
          <aside className="rail">
            <section className="card">
              <div className="card-head">
                <h2>Leaf photo</h2>
                <ImagePlus size={20} />
              </div>

              {!file ? (
                <div
                  className={`dropzone ${dragging ? "is-dragging" : ""}`}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  role="button"
                  tabIndex={0}
                  aria-label="Upload a leaf photo"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      fileInputRef.current?.click();
                    }
                  }}
                >
                  <span className="dropzone-icon">
                    <Upload size={24} />
                  </span>
                  <strong>Drop a photo here</strong>
                  <span className="dropzone-sub">or click to browse your files</span>
                  <span className="dropzone-hint">
                    JPG, PNG, WEBP or BMP, up to 10 MB
                  </span>
                </div>
              ) : (
                <div className="file-card">
                  <img src={preview} alt="" className="file-thumb" />
                  <div className="file-meta">
                    <strong title={file.name}>{file.name}</strong>
                    <span>{formatSize(file.size)}</span>
                  </div>
                  <button
                    className="icon-button"
                    onClick={clearImage}
                    aria-label="Remove selected image"
                    type="button"
                    disabled={loading}
                  >
                    <X size={18} />
                  </button>
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
                <div className="notice notice-error" role="alert">
                  <CircleAlert size={18} />
                  <span>{error}</span>
                </div>
              )}

              <button
                className="primary-button"
                onClick={analyseImage}
                disabled={!file || loading}
                type="button"
              >
                {loading ? (
                  <>
                    <LoaderCircle className="spin" size={19} />
                    Analysing
                  </>
                ) : (
                  <>
                    <ScanSearch size={19} />
                    {result ? "Analyse again" : "Analyse leaf"}
                  </>
                )}
              </button>

              <p className="privacy">
                <ShieldCheck size={16} />
                Your photo stays on this computer. It goes only to your local API.
              </p>
            </section>

            <section className="card card-quiet">
              <h2 className="card-title-sm">How to read the results</h2>
              <dl className="guide">
                <div>
                  <dt>
                    <span className="swatch swatch-box" />
                    Green boxes
                  </dt>
                  <dd>Regions the trained YOLO model flagged, with a confidence score.</dd>
                </div>
                <div>
                  <dt>
                    <span className="swatch swatch-fill" />
                    Red areas
                  </dt>
                  <dd>
                    Candidate regions from colour thresholds. Shadows and veins
                    can be included.
                  </dd>
                </div>
              </dl>
            </section>
          </aside>

          {/* ---------- Right: stage + findings ---------- */}
          <section className="results" aria-live="polite">
            <div className="stage">
              <div className="stage-bar">
                {result ? (
                  <div className="tabs" role="tablist" aria-label="Analysis views">
                    {VIEWS.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        role="tab"
                        aria-selected={activeView === item.id}
                        className={`tab ${activeView === item.id ? "is-active" : ""}`}
                        onClick={() => selectView(item.id)}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                ) : (
                  <span className="stage-title">
                    {loading
                      ? "Analysing your leaf"
                      : preview
                        ? "Ready to analyse"
                        : "Waiting for a photo"}
                  </span>
                )}

                {result && imageSrc && (
                  <a
                    className="stage-action"
                    href={imageSrc}
                    download={downloadName}
                  >
                    <Download size={16} />
                    Save image
                  </a>
                )}
              </div>

              <div className={`viewport ${loading ? "is-scanning" : ""}`}>
                <span className="corner corner-tl" />
                <span className="corner corner-tr" />
                <span className="corner corner-bl" />
                <span className="corner corner-br" />

                {imageSrc ? (
                  <img
                    key={`${activeView}-${result?.filename ?? preview}`}
                    className="stage-image"
                    src={imageSrc}
                    alt={result ? IMAGE_ALT[activeView] : IMAGE_ALT.original}
                    onError={() =>
                      setImageError(
                        `The ${view.label.toLowerCase()} image could not be displayed. Check that the API returned valid Base64 PNG data.`
                      )
                    }
                    onLoad={() => setImageError("")}
                  />
                ) : result ? (
                  <div className="stage-empty">
                    <CircleAlert size={26} />
                    <p>The API did not include this view in its response.</p>
                  </div>
                ) : (
                  <div className="stage-empty">
                    <svg
                      className="leaf-outline"
                      viewBox="0 0 120 120"
                      aria-hidden="true"
                    >
                      <path
                        d="M60 108C60 108 18 88 18 52C18 30 36 14 60 10C84 14 102 30 102 52C102 88 60 108 60 108Z"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <path
                        d="M60 108V28M60 46L40 34M60 62L36 48M60 78L42 66M60 46L80 34M60 62L84 48M60 78L78 66"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                    </svg>
                    <h3>Upload a leaf photo to begin</h3>
                    <p>
                      The original, detection and segmentation views appear here
                      after the analysis.
                    </p>
                  </div>
                )}

                {loading && <span className="scan-line" aria-hidden="true" />}
              </div>

              {result && view.legend.length > 0 && (
                <ul className="legend">
                  {view.legend.map((item) => (
                    <li key={item.text}>
                      <span className={`swatch swatch-${item.kind}`} />
                      {item.text}
                    </li>
                  ))}
                </ul>
              )}

              {imageError && (
                <div className="notice notice-error notice-on-dark" role="alert">
                  <CircleAlert size={18} />
                  <span>{imageError}</span>
                </div>
              )}
            </div>

            {result && (
              <>
                {activeView === "segmentation" && (
                  <div className="notice notice-warn">
                    <CircleAlert size={18} />
                    <span>
                      Red pixels are heuristic candidate regions, not confirmed
                      lesions. Shadows, veins and healthy tissue may also be
                      included.
                    </span>
                  </div>
                )}

                <div className="metrics">
                  <div className="metric">
                    <span className="metric-label">Regions flagged by YOLO</span>
                    <strong className="metric-value">{detectionCount}</strong>
                  </div>

                  {stats && (
                    <div className="metric">
                      <span className="metric-label">Candidate area</span>
                      <strong className="metric-value">
                        {stats.candidate_ratio_percent ?? 0}
                        <span className="metric-unit">%</span>
                      </strong>
                      <span className="metric-sub">
                        {Number(stats.candidate_region_pixels ?? 0).toLocaleString()}{" "}
                        pixels
                      </span>
                    </div>
                  )}

                  <div className="metric">
                    <span className="metric-label">Image size</span>
                    <strong className="metric-value metric-value-sm">
                      {width ?? "—"} × {height ?? "—"}
                      <span className="metric-unit"> px</span>
                    </strong>
                    <span className="metric-sub" title={result.filename ?? file?.name}>
                      {result.filename ?? file?.name ?? "Uploaded image"}
                    </span>
                  </div>
                </div>

                <section className="card">
                  <div className="card-head">
                    <h2>Detected regions</h2>
                  </div>

                  {!result.detections?.length ? (
                    <div className="notice notice-warn">
                      <CircleAlert size={18} />
                      <span>
                        No regions passed the current confidence threshold. That
                        does not mean the leaf is healthy.
                      </span>
                    </div>
                  ) : (
                    <ul className="detections">
                      {result.detections.map((detection, index) => {
                        const pct = Math.max(
                          0,
                          Math.min(100, detection.confidence * 100)
                        );
                        return (
                          <li
                            className="detection"
                            key={`${index}-${detection.class_id}`}
                          >
                            <span className="detection-index">{index + 1}</span>
                            <div className="detection-body">
                              <div className="detection-top">
                                <strong>{detection.class_name}</strong>
                                <span className="detection-pct">
                                  {pct.toFixed(1)}%
                                </span>
                              </div>
                              <div
                                className="bar"
                                role="progressbar"
                                aria-valuenow={Math.round(pct)}
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-label={`Confidence for region ${index + 1}`}
                              >
                                <span style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>

                <div className="notice notice-warn">
                  <ShieldCheck size={18} />
                  <span>
                    {result.notice ||
                      "This is experimental model output, not a confirmed diagnosis. Have a specialist verify it."}
                  </span>
                </div>
              </>
            )}
          </section>
        </div>

        <footer className="footer">
          <span>Potato Vision is a research prototype.</span>
          <span>YOLO and OpenCV, running locally.</span>
        </footer>
      </main>
    </div>
  );
}

export default App;