// ============================================================
// AuraVision Dashboard Controller
// AI Image Quality & Defect Detection
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    // ============================================================
    // API CONFIG
    // ============================================================

    const API_BASE = window.location.origin;

    // ============================================================
    // APPLICATION STATE
    // ============================================================

    let currentResult = null;

    // ============================================================
    // DOM ELEMENTS
    // ============================================================

    const systemStatus =
        document.getElementById("systemStatus");

    const dropZone =
        document.getElementById("dropZone");

    const fileInput =
        document.getElementById("fileInput");

    const uploadProgress =
        document.getElementById("uploadProgress");

    const resultsSection =
        document.getElementById("resultsSection");

    const historyList =
        document.getElementById("historyList");

    const btnRefreshHistory =
        document.getElementById("btnRefreshHistory");

    const toast =
        document.getElementById("toast");

    const toastMsg =
        document.getElementById("toastMsg");

    // ============================================================
    // SCORE ELEMENTS
    // ============================================================

    const scoreProgress =
        document.getElementById("scoreProgress");

    const scoreValue =
        document.getElementById("scoreValue");

    const assessmentBadge =
        document.getElementById("assessmentBadge");

    const assessmentDesc =
        document.getElementById("assessmentDesc");

    // ============================================================
    // IMAGE VIEWER
    // ============================================================

    const btnOriginal =
        document.getElementById("btnOriginal");

    const btnHeatmap =
        document.getElementById("btnHeatmap");

    const displayOriginal =
        document.getElementById("displayOriginal");

    const displayHeatmap =
        document.getElementById("displayHeatmap");

    const heatmapInfo =
        document.getElementById("heatmapInfo");

    // ============================================================
    // ISSUES
    // ============================================================

    const issuesList =
        document.getElementById("issuesList");

    // ============================================================
    // STATISTICS
    // ============================================================

    const statSharpness =
        document.getElementById("statSharpness");

    const barSharpness =
        document.getElementById("barSharpness");

    const statBrightness =
        document.getElementById("statBrightness");

    const barBrightness =
        document.getElementById("barBrightness");

    const statContrast =
        document.getElementById("statContrast");

    const barContrast =
        document.getElementById("barContrast");

    const statNoise =
        document.getElementById("statNoise");

    const barNoise =
        document.getElementById("barNoise");

    const statFFT =
        document.getElementById("statFFT");

    const barFFT =
        document.getElementById("barFFT");


    // ============================================================
    // HTML ESCAPE
    // ============================================================
    // NOTE: This was missing from the original file even though
    // renderIssues() and renderHistory() call it repeatedly.
    // Without this, every render call throws a ReferenceError
    // and rendering silently breaks. Added here so it's defined
    // before any function that uses it.

    function escapeHtml(str) {

        if (str === null || str === undefined) {
            return "";
        }

        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    // ============================================================
    // INITIALIZE
    // ============================================================

    checkBackendHealth();
    loadHistory();


    // ============================================================
    // BACKEND HEALTH
    // ============================================================

    async function checkBackendHealth() {

        try {

            const response =
                await fetch(`${API_BASE}/health`, {
                    cache: "no-store"
                });

            if (!response.ok) {
                setBackendOffline();
                return;
            }

            const data =
                await response.json();

            const indicator =
                systemStatus?.querySelector(
                    ".status-indicator"
                );

            const text =
                systemStatus?.querySelector(
                    ".status-text"
                );

            if (!indicator || !text) {
                return;
            }

            indicator.className =
                "status-indicator online";

            if (
                data.models &&
                data.models.mode === "heuristic_fallback"
            ) {

                text.textContent =
                    "Running in Heuristic Mode";

                text.title =
                    "ML models are unavailable. Using computer vision fallback.";

            } else {

                text.textContent =
                    "AI System Online";

                text.title =
                    "Hybrid ML pipeline is fully operational.";
            }

        } catch (error) {

            console.error(
                "Backend health check failed:",
                error
            );

            setBackendOffline();
        }
    }


    // ============================================================
    // BACKEND OFFLINE
    // ============================================================

    function setBackendOffline() {

        const indicator =
            systemStatus?.querySelector(
                ".status-indicator"
            );

        const text =
            systemStatus?.querySelector(
                ".status-text"
            );

        if (indicator) {

            indicator.className =
                "status-indicator offline";
        }

        if (text) {

            text.textContent =
                "Backend Offline";
        }

        showToast(
            "Cannot connect to server. Make sure FastAPI is running."
        );
    }


    // ============================================================
    // DRAG & DROP
    // ============================================================

    ["dragenter", "dragover"].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();
                    event.stopPropagation();

                    dropZone.classList.add(
                        "dragover"
                    );
                }
            );
        }
    );


    ["dragleave", "drop"].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();
                    event.stopPropagation();

                    dropZone.classList.remove(
                        "dragover"
                    );
                }
            );
        }
    );


    dropZone.addEventListener(
        "drop",
        event => {

            const files =
                event.dataTransfer?.files;

            if (
                files &&
                files.length > 0
            ) {

                handleFileUpload(
                    files[0]
                );
            }
        }
    );


    dropZone.addEventListener(
        "click",
        event => {

            /*
             * Prevent double opening when clicking
             * the Browse Files label.
             */

            if (
                event.target.tagName === "LABEL" ||
                event.target.closest("label")
            ) {
                return;
            }

            fileInput.click();
        }
    );


    fileInput.addEventListener(
        "change",
        event => {

            if (
                event.target.files &&
                event.target.files.length > 0
            ) {

                handleFileUpload(
                    event.target.files[0]
                );
            }

            /*
             * Allows selecting the same file again.
             */
            fileInput.value = "";
        }
    );


    // ============================================================
    // FILE UPLOAD
    // ============================================================

    async function handleFileUpload(file) {

        if (!file) {
            return;
        }


        // --------------------------------------------------------
        // SIZE VALIDATION
        // --------------------------------------------------------

        if (
            file.size >
            10 * 1024 * 1024
        ) {

            showToast(
                "File is too large. Maximum size is 10MB."
            );

            return;
        }


        // --------------------------------------------------------
        // TYPE VALIDATION
        // --------------------------------------------------------

        const allowedTypes = [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp"
        ];

        if (
            !allowedTypes.includes(
                file.type
            )
        ) {

            showToast(
                "Unsupported format. Please upload JPEG, PNG or WebP."
            );

            return;
        }


        // --------------------------------------------------------
        // LOADING UI
        // --------------------------------------------------------

        dropZone.classList.add(
            "hidden"
        );

        uploadProgress.classList.remove(
            "hidden"
        );

        resultsSection.classList.add(
            "hidden"
        );


        // --------------------------------------------------------
        // FORM DATA
        // --------------------------------------------------------

        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        // --------------------------------------------------------
        // SEND TO FASTAPI
        // --------------------------------------------------------

        try {

            const response =
                await fetch(
                    `${API_BASE}/api/analyze`,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            if (!response.ok) {

                let errorMessage =
                    "Server error occurred during analysis.";

                try {

                    const errorData =
                        await response.json();

                    if (
                        errorData &&
                        errorData.detail
                    ) {

                        errorMessage =
                            errorData.detail;
                    }

                } catch (parseError) {

                    console.error(
                        "Error parsing server response:",
                        parseError
                    );
                }

                throw new Error(
                    errorMessage
                );
            }


            // ----------------------------------------------------
            // PARSE RESULT
            // ----------------------------------------------------

            const data =
                await response.json();

            console.log(
                "NEW ANALYSIS RESULT:",
                data
            );


            // ----------------------------------------------------
            // STORE CURRENT RESULT
            // ----------------------------------------------------

            currentResult =
                data;


            // ----------------------------------------------------
            // HIDE LOADING
            // ----------------------------------------------------

            uploadProgress.classList.add(
                "hidden"
            );

            dropZone.classList.remove(
                "hidden"
            );

            resultsSection.classList.remove(
                "hidden"
            );


            // ----------------------------------------------------
            // RENDER RESULT
            // ----------------------------------------------------

            renderResult(
                data
            );


            // ----------------------------------------------------
            // REFRESH HISTORY
            // ----------------------------------------------------

            await loadHistory();

        } catch (error) {

            console.error(
                "Upload error:",
                error
            );

            uploadProgress.classList.add(
                "hidden"
            );

            dropZone.classList.remove(
                "hidden"
            );

            showToast(
                error.message ||
                "Failed to analyze image."
            );
        }
    }


    // ============================================================
    // URL BUILDER
    // ============================================================

    function buildAssetUrl(url) {

        if (!url) {
            return null;
        }

        let cleanUrl =
            String(url).trim();

        if (!cleanUrl) {
            return null;
        }


        // --------------------------------------------------------
        // Already absolute
        // --------------------------------------------------------

        if (
            cleanUrl.startsWith("http://") ||
            cleanUrl.startsWith("https://")
        ) {

            return cleanUrl;
        }


        // --------------------------------------------------------
        // Backend already gives /static/...
        // --------------------------------------------------------

        if (
            cleanUrl.startsWith("/")
        ) {

            return `${API_BASE}${cleanUrl}`;
        }


        // --------------------------------------------------------
        // Windows path
        // --------------------------------------------------------

        cleanUrl =
            cleanUrl.replace(
                /\\/g,
                "/"
            );


        /*
         * Examples:
         *
         * static/uploads/file.jpg
         * static/heatmaps/file.jpg
         * uploads/file.jpg
         * heatmaps/file.jpg
         */


        if (
            cleanUrl.startsWith(
                "static/"
            )
        ) {

            return `${API_BASE}/${cleanUrl}`;
        }


        if (
            cleanUrl.startsWith(
                "uploads/"
            )
        ) {

            return `${API_BASE}/static/${cleanUrl}`;
        }


        if (
            cleanUrl.startsWith(
                "heatmaps/"
            )
        ) {

            return `${API_BASE}/static/${cleanUrl}`;
        }


        /*
         * Last fallback.
         *
         * IMPORTANT:
         * We do NOT automatically assume this is
         * the original filename.
         *
         * This is only for old database records
         * where the backend stored a bare UUID filename.
         */

        return `${API_BASE}/static/uploads/${encodeURIComponent(
            cleanUrl
        )}`;
    }


    // ============================================================
    // GET ORIGINAL IMAGE URL
    // ============================================================

    function getOriginalImageUrl(result) {

        /*
         * NEW CORRECT FIELD:
         *
         * /static/uploads/UUID.jpg
         */

        if (
            result &&
            result.original_image_url
        ) {

            return buildAssetUrl(
                result.original_image_url
            );
        }


        /*
         * Compatibility with older API/database records.
         */

        if (
            result &&
            result.original_relative_path
        ) {

            return buildAssetUrl(
                result.original_relative_path
            );
        }


        if (
            result &&
            result.original_image_path
        ) {

            return buildAssetUrl(
                result.original_image_path
            );
        }


        /*
         * Older frontend/backend records may only have
         * filename.
         *
         * This will work ONLY if filename itself is
         * the physical UUID filename.
         */

        if (
            result &&
            result.filename
        ) {

            const filename =
                String(
                    result.filename
                );

            /*
             * UUID-based filenames have this pattern.
             */

            const uuidPattern =
                /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.[a-z0-9]+$/i;

            if (
                uuidPattern.test(
                    filename
                )
            ) {

                return buildAssetUrl(
                    `/static/uploads/${filename}`
                );
            }
        }


        return null;
    }


    // ============================================================
    // GET HEATMAP URL
    // ============================================================

    function getHeatmapImageUrl(result) {

        if (
            !result ||
            !result.heatmap_url
        ) {

            return null;
        }

        return buildAssetUrl(
            result.heatmap_url
        );
    }


    // ============================================================
    // RENDER RESULT
    // ============================================================

    function renderResult(result) {

        console.log(
            "Rendering result:",
            result
        );


        if (!result) {
            return;
        }


        // ========================================================
        // QUALITY LABEL
        // ========================================================

        assessmentBadge.className =
            "assessment-badge";


        let colorClass =
            "var(--warning)";

        let descText =
            "Image quality assessment completed.";


        const label =
            String(
                result.quality_label ||
                "DEGRADED"
            ).toUpperCase();


        if (
            label === "ACCEPTABLE"
        ) {

            assessmentBadge.classList.add(
                "badge-acceptable"
            );

            assessmentBadge.textContent =
                "ACCEPTABLE";

            descText =
                "The image meets baseline clarity and exposure requirements.";

            colorClass =
                "var(--success)";

        } else if (
            label === "DEGRADED"
        ) {

            assessmentBadge.classList.add(
                "badge-degraded"
            );

            assessmentBadge.textContent =
                "DEGRADED";

            descText =
                "Minor sharpness, noise, or exposure anomalies identified.";

            colorClass =
                "var(--warning)";

        } else {

            assessmentBadge.classList.add(
                "badge-defective"
            );

            assessmentBadge.textContent =
                "DEFECTIVE";

            descText =
                "Severe optical degradation, noise corruption, or physical surface defects detected.";

            colorClass =
                "var(--danger)";
        }


        assessmentDesc.textContent =
            descText;


        // ========================================================
        // SCORE
        // ========================================================

        const targetScore =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(
                        result.quality_score
                    ) || 0
                )
            );


        scoreProgress.style.setProperty(
            "--progress-color",
            colorClass
        );


        /*
         * Cancel previous animation if necessary.
         */

        if (
            scoreProgress._animationTimer
        ) {

            clearInterval(
                scoreProgress._animationTimer
            );
        }


        let currentScore = 0;

        scoreValue.textContent =
            "0";

        scoreProgress.style.setProperty(
            "--progress-val",
            "0"
        );


        scoreProgress._animationTimer =
            setInterval(
                () => {

                    if (
                        currentScore >=
                        targetScore
                    ) {

                        clearInterval(
                            scoreProgress._animationTimer
                        );

                        scoreValue.textContent =
                            targetScore.toFixed(
                                targetScore % 1 === 0
                                    ? 0
                                    : 1
                            );

                        scoreProgress.style.setProperty(
                            "--progress-val",
                            targetScore
                        );

                        return;
                    }


                    currentScore += Math.max(
                        1,
                        Math.ceil(
                            (
                                targetScore -
                                currentScore
                            ) / 5
                        )
                    );


                    if (
                        currentScore >
                        targetScore
                    ) {

                        currentScore =
                            targetScore;
                    }


                    scoreValue.textContent =
                        currentScore;


                    scoreProgress.style.setProperty(
                        "--progress-val",
                        currentScore
                    );

                },
                30
            );


        // ========================================================
        // ORIGINAL IMAGE
        // ========================================================

        const originalUrl =
            getOriginalImageUrl(
                result
            );


        console.log(
            "Resolved original image URL:",
            originalUrl
        );


        /*
         * Remove old handlers first.
         */

        displayOriginal.onload =
            null;

        displayOriginal.onerror =
            null;


        if (originalUrl) {

            displayOriginal.src =
                originalUrl;


            displayOriginal.onload =
                () => {

                    console.log(
                        "Original image loaded:",
                        originalUrl
                    );
                };


            displayOriginal.onerror =
                () => {

                    console.error(
                        "Original image FAILED:",
                        originalUrl
                    );

                    showToast(
                        "Original image could not be loaded."
                    );
                };

        } else {

            displayOriginal.removeAttribute(
                "src"
            );

            console.warn(
                "No original image URL available."
            );
        }


        // ========================================================
        // HEATMAP
        // ========================================================

        const heatmapUrl =
            getHeatmapImageUrl(
                result
            );


        console.log(
            "Resolved heatmap URL:",
            heatmapUrl
        );


        displayHeatmap.onload =
            null;

        displayHeatmap.onerror =
            null;


        if (heatmapUrl) {

            displayHeatmap.src =
                heatmapUrl;


            displayHeatmap.onload =
                () => {

                    console.log(
                        "Heatmap loaded:",
                        heatmapUrl
                    );
                };


            displayHeatmap.onerror =
                () => {

                    console.error(
                        "Heatmap FAILED:",
                        heatmapUrl
                    );
                };


            btnHeatmap.classList.remove(
                "hidden"
            );

        } else {

            displayHeatmap.removeAttribute(
                "src"
            );

            btnHeatmap.classList.add(
                "hidden"
            );
        }


        // ========================================================
        // DEFAULT TAB
        // ========================================================

        showTab(
            "original"
        );


        // ========================================================
        // ISSUES
        // ========================================================

        renderIssues(
            result.issues
        );


        // ========================================================
        // FEATURES
        // ========================================================

        renderFeatures(
            result.features
        );
    }


    // ============================================================
    // RENDER ISSUES
    // ============================================================

    function renderIssues(issues) {

        issuesList.innerHTML =
            "";


        if (
            !Array.isArray(issues) ||
            issues.length === 0
        ) {

            issuesList.innerHTML = `
                <div class="empty-issues">
                    <i class="fa-solid fa-circle-check text-success"></i>
                    <p>No quality defects detected. Image is clear.</p>
                </div>
            `;

            return;
        }


        issues.forEach(
            issue => {

                const div =
                    document.createElement(
                        "div"
                    );


                const type =
                    String(
                        issue.type ||
                        "unknown"
                    );


                const severity =
                    String(
                        issue.severity ||
                        "low"
                    ).toLowerCase();


                const confidence =
                    Math.max(
                        0,
                        Math.min(
                            1,
                            Number(
                                issue.confidence
                            ) || 0
                        )
                    );


                const isSevere =
                    severity === "high";


                div.className =
                    `issue-item ${isSevere
                        ? "issue-high"
                        : "issue-low"
                    }`;


                let description =
                    "Image quality issue detected.";


                if (
                    type === "blur"
                ) {

                    description =
                        "Insufficient sharpness or focus blur.";

                } else if (
                    type === "underexposure"
                ) {

                    description =
                        "Under-illuminated or shadow clipping.";

                } else if (
                    type === "overexposure"
                ) {

                    description =
                        "Over-saturated brightness or highlight clipping.";

                } else if (
                    type === "noise"
                ) {

                    description =
                        "High sensor noise or luminance grain.";

                } else if (
                    type === "corruption"
                ) {

                    description =
                        "Compression artifacts or image corruption detected.";

                } else if (
                    type === "defect"
                ) {

                    const subtype =
                        issue.subtype ||
                        "physical anomaly";

                    description =
                        `Surface anomaly: ${subtype}.`;
                }


                const displayType =
                    type === "defect"
                        ? (
                            issue.subtype ||
                            "defect"
                        )
                        : type;


                div.innerHTML = `

                    <div class="issue-details">

                        <span class="issue-type">
                            ${escapeHtml(
                    displayType
                )}
                        </span>

                        <span class="issue-meta">
                            ${escapeHtml(
                    description
                )}
                        </span>

                    </div>

                    <div class="issue-confidence">

                        <span class="confidence-val">
                            ${(
                        confidence *
                        100
                    ).toFixed(0)}%
                        </span>

                        <span class="confidence-label">
                            Confidence
                        </span>

                    </div>
                `;


                issuesList.appendChild(
                    div
                );
            }
        );
    }


    // ============================================================
    // RENDER FEATURES
    // ============================================================

    function renderFeatures(features) {

        const f =
            features || {};


        // --------------------------------------------------------
        // SHARPNESS
        // --------------------------------------------------------

        const sharpness =
            Number(
                f.blur_laplacian_var
            ) || 0;


        statSharpness.textContent =
            sharpness.toFixed(1);


        const sharpnessPct =
            Math.min(
                100,
                Math.max(
                    0,
                    (
                        sharpness /
                        220
                    ) * 100
                )
            );


        barSharpness.style.width =
            `${sharpnessPct}%`;


        // --------------------------------------------------------
        // BRIGHTNESS
        // --------------------------------------------------------

        const brightness =
            Number(
                f.brightness_mean
            ) || 0;


        statBrightness.textContent =
            brightness.toFixed(1);


        const brightnessPct =
            Math.min(
                100,
                Math.max(
                    0,
                    (
                        brightness /
                        255
                    ) * 100
                )
            );


        barBrightness.style.width =
            `${brightnessPct}%`;


        // --------------------------------------------------------
        // CONTRAST
        // --------------------------------------------------------

        const contrast =
            Number(
                f.contrast_rms
            ) || 0;


        statContrast.textContent =
            contrast.toFixed(1);


        const contrastPct =
            Math.min(
                100,
                Math.max(
                    0,
                    (
                        contrast /
                        90
                    ) * 100
                )
            );


        barContrast.style.width =
            `${contrastPct}%`;


        // --------------------------------------------------------
        // NOISE
        // --------------------------------------------------------

        const noise =
            Number(
                f.noise_std
            ) || 0;


        statNoise.textContent =
            noise.toFixed(2);


        const noisePct =
            Math.min(
                100,
                Math.max(
                    0,
                    (
                        noise /
                        20
                    ) * 100
                )
            );


        barNoise.style.width =
            `${noisePct}%`;


        if (
            noise > 8
        ) {

            barNoise.parentElement.classList.add(
                "bg-danger"
            );

        } else {

            barNoise.parentElement.classList.remove(
                "bg-danger"
            );
        }


        // --------------------------------------------------------
        // FFT
        // --------------------------------------------------------

        const fft =
            Number(
                f.fft_high_freq_ratio
            ) || 0;


        statFFT.textContent =
            `${(
                fft * 100
            ).toFixed(2)}%`;


        const fftPct =
            Math.min(
                100,
                Math.max(
                    0,
                    fft * 100
                )
            );


        barFFT.style.width =
            `${fftPct}%`;
    }


    // ============================================================
    // IMAGE TAB SWITCHER
    // ============================================================

    function showTab(tabName) {

        if (
            tabName === "original"
        ) {

            btnOriginal.classList.add(
                "active"
            );

            btnHeatmap.classList.remove(
                "active"
            );

            displayOriginal.classList.remove(
                "hidden"
            );

            displayHeatmap.classList.add(
                "hidden"
            );

            heatmapInfo.classList.add(
                "hidden"
            );

            return;
        }


        /*
         * Don't allow heatmap tab if no heatmap exists.
         */

        if (
            btnHeatmap.classList.contains(
                "hidden"
            )
        ) {

            return;
        }


        btnOriginal.classList.remove(
            "active"
        );

        btnHeatmap.classList.add(
            "active"
        );

        displayOriginal.classList.add(
            "hidden"
        );

        displayHeatmap.classList.remove(
            "hidden"
        );

        heatmapInfo.classList.remove(
            "hidden"
        );
    }


    btnOriginal.addEventListener(
        "click",
        () => {

            showTab(
                "original"
            );
        }
    );


    btnHeatmap.addEventListener(
        "click",
        () => {

            showTab(
                "heatmap"
            );
        }
    );


    // ============================================================
    // LOAD HISTORY
    // ============================================================

    async function loadHistory() {

        try {

            const response =
                await fetch(
                    `${API_BASE}/api/results?limit=25`,
                    {
                        cache: "no-store"
                    }
                );


            if (!response.ok) {

                console.error(
                    "History request failed:",
                    response.status
                );

                return;
            }


            const list =
                await response.json();


            console.log(
                "History loaded:",
                list
            );


            renderHistory(
                list
            );

        } catch (error) {

            console.error(
                "Error loading history:",
                error
            );
        }
    }


    // ============================================================
    // RENDER HISTORY
    // ============================================================

    function renderHistory(items) {

        historyList.innerHTML =
            "";


        if (
            !Array.isArray(items) ||
            items.length === 0
        ) {

            historyList.innerHTML = `
                <div class="history-empty">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>No previous uploads found.</p>
                </div>
            `;

            return;
        }


        items.forEach(
            item => {

                const div =
                    document.createElement(
                        "div"
                    );


                const label =
                    String(
                        item.quality_label ||
                        "DEGRADED"
                    );


                const labelClass =
                    label.toLowerCase();


                const isActive =
                    currentResult &&
                    Number(
                        currentResult.id
                    ) ===
                    Number(
                        item.id
                    );


                div.className =
                    `history-item ${isActive
                        ? "active"
                        : ""
                    }`;


                // ------------------------------------------------
                // TIMESTAMP
                // ------------------------------------------------

                const date =
                    item.timestamp
                        ? new Date(
                            item.timestamp
                        )
                        : null;


                let dateStr =
                    "Unknown date";

                let timeStr =
                    "";


                if (
                    date &&
                    !isNaN(
                        date.getTime()
                    )
                ) {

                    dateStr =
                        date.toLocaleDateString(
                            [],
                            {
                                month: "short",
                                day: "numeric"
                            }
                        );


                    timeStr =
                        date.toLocaleTimeString(
                            [],
                            {
                                hour: "2-digit",
                                minute: "2-digit"
                            }
                        );
                }


                // ------------------------------------------------
                // HISTORY HTML
                // ------------------------------------------------

                div.innerHTML = `

                    <div
                        class="history-thumbnail-placeholder ${escapeHtml(
                    labelClass
                )}"
                    >
                        ${Number(
                    item.quality_score || 0
                ).toFixed(0)}
                    </div>

                    <div class="history-info">

                        <span
                            class="history-name"
                            title="${escapeHtml(
                    item.filename ||
                    "Unknown"
                )}"
                        >
                            ${escapeHtml(
                    item.filename ||
                    "Unknown"
                )}
                        </span>

                        <span class="history-meta">

                            <span>
                                ${escapeHtml(
                    label
                )}
                            </span>

                            <span>
                                ${escapeHtml(
                    dateStr
                )}
                                ${escapeHtml(
                    timeStr
                        ? ", " + timeStr
                        : ""
                )}
                            </span>

                        </span>

                    </div>
                `;


                // ------------------------------------------------
                // CLICK HISTORY
                // ------------------------------------------------

                div.addEventListener(
                    "click",
                    () => {

                        document
                            .querySelectorAll(
                                ".history-item"
                            )
                            .forEach(
                                element => {

                                    element.classList.remove(
                                        "active"
                                    );
                                }
                            );


                        div.classList.add(
                            "active"
                        );


                        currentResult =
                            item;


                        resultsSection.classList.remove(
                            "hidden"
                        );


                        renderResult(
                            item
                        );


                        resultsSection.scrollIntoView(
                            {
                                behavior: "smooth",
                                block: "start"
                            }
                        );
                    }
                );


                historyList.appendChild(
                    div
                );
            }
        );
    }


    // ============================================================
    // REFRESH HISTORY
    // ============================================================

    if (
        btnRefreshHistory
    ) {

        btnRefreshHistory.addEventListener(
            "click",
            async () => {

                btnRefreshHistory.disabled =
                    true;

                try {

                    await loadHistory();

                } finally {

                    btnRefreshHistory.disabled =
                        false;
                }
            }
        );
    }


    // ============================================================
    // TOAST
    // ============================================================

    let toastTimer = null;


    function showToast(message) {

        if (
            !toast ||
            !toastMsg
        ) {

            return;
        }


        toastMsg.textContent =
            message;


        toast.classList.remove(
            "hidden"
        );


        if (
            toastTimer
        ) {

            clearTimeout(
                toastTimer
            );
        }


        toastTimer =
            setTimeout(
                () => {

                    toast.classList.add(
                        "hidden"
                    );
                },
                4000
            );
    }

});