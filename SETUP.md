# Setup Guide

This guide explains how to install and run the Music Assistant Server manually.

## Requirements
- **Python 3.12 or newer**
- **ffmpeg** version 6.1 or later must be available on your system.
- **uv** package manager for installing Python dependencies.

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/music-assistant/server.git
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade `pip` and install `uv`:
   ```bash
   pip install --upgrade pip uv
   ```
4. Install Music Assistant and its dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```
   To enable the optional Music Insights provider run:
   ```bash
   uv pip install -e .[music-insights]
   ```

## Running the Server
Start the server with the `mass` command:
```bash
mass
```
By default the configuration and database files are stored in `~/.musicassistant`. You can specify a different directory with:
```bash
mass --config /path/to/config
```
Additional useful options:
- `--log-level <level>` – set logging verbosity (`info`, `warning`, `debug`, `verbose`, ...).
- `--safe-mode` – start without loading any providers.

## Next Steps
Once the server is running, open the web interface on `http://<server-ip>:8095` and follow the onboarding steps to add providers and players.
