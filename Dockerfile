# Upwork job-alert agent - FastAPI service.
#
# The pipeline opens real Chrome (SeleniumBase UC mode) and solves the Cloudflare
# Turnstile with a PyAutoGUI mouse click, which needs a display. In a Linux
# container that means Chrome + Xvfb (a virtual display) + the X libs PyAutoGUI
# uses. Set browser_headless: true and browser_xvfb: true in config.yaml when
# running here (SeleniumBase starts the Xvfb display itself when xvfb=True).

# Pinned to amd64 + Debian bookworm: Google Chrome only ships a linux/amd64 .deb,
# so on an Apple Silicon (arm64) host the image must be built for amd64 (runs under
# emulation locally, native on an amd64 server). bookworm keeps the classic package
# names Chrome depends on (e.g. libasound2, not libasound2t64). On an amd64 host you
# can drop the --platform for a native build.
FROM --platform=linux/amd64 python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps: Chrome runtime libs, Xvfb + X utils for the virtual display, and
# scrot (PyAutoGUI screenshotting used by the captcha solve).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates \
        xvfb xauth x11-utils scrot \
        fonts-liberation libnss3 libxss1 libasound2 libatk-bridge2.0-0 \
        libgtk-3-0 libgbm1 libu2f-udev \
    && wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm -f /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + config.
COPY config.yaml .
COPY resources/ ./resources/
COPY src/ ./src/

# Persisted SQLite store lives here; mount a volume to keep it across restarts.
VOLUME ["/app/resources"]

ENV POLL_INTERVAL_SECONDS=300 \
    PERSIST_JOBS=1
EXPOSE 8000

WORKDIR /app/src
# Serve the FastAPI layer. To run the original polling loop instead, override:
#   docker run ... python main.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
