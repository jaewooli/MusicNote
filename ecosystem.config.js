// pm2 process definition for MusicNote.
//   pm2 start ecosystem.config.js
//   pm2 logs musicnote
//   pm2 restart musicnote
const path = require('path');
const ROOT = __dirname;

// The GPU worker is NOT configured here. deploy/vast/gpu.sh writes its URL to
// deploy/vast/current-url and mt3_bridge reads that file per request, so
// switching between GPU and CPU needs no restart — and does not wipe JOBS.

module.exports = {
  apps: [
    {
      name: 'musicnote',
      // run uvicorn from the project virtualenv
      script: path.join(ROOT, '.venv/bin/uvicorn'),
      args: 'app:app --host 127.0.0.1 --port 8731 --workers 1 --timeout-keep-alive 65',
      cwd: path.join(ROOT, 'backend'),
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      kill_timeout: 8000,
      max_memory_restart: '4000M',                 // Demucs (stems mode) peaks ~1.8 GB
      env: {
        PYTHONUNBUFFERED: '1',
        MUSICNOTE_WORKDIR: path.join(ROOT, 'uploads'),
        MUSICNOTE_MAX_MB: '40',
        MUSICNOTE_MAX_DURATION: '1200',            // YouTube: max 20 min
        MUSICNOTE_STEMS_MAX_DURATION: '300',       // stems mode: max 5 min (slow on CPU)
        OMP_NUM_THREADS: '4',
        // YouTube blocks datacenter IPs; point this at a cookies.txt exported
        // from a logged-in browser to enable URL input. Defaults to
        // backend/cookies.txt if that file exists.
        MUSICNOTE_YT_COOKIES: path.join(ROOT, 'backend/cookies.txt'),
        // JS runtime (Deno) + bgutil PO-token provider, used automatically by yt-dlp
        PATH: [
          path.join(process.env.HOME || '/home/ubuntu', '.local/deno/bin'),
          process.env.PATH,
        ].join(':'),
        MUSICNOTE_POT_BASEURL: 'http://127.0.0.1:4416',
      },
      out_file: path.join(ROOT, 'logs/out.log'),
      error_file: path.join(ROOT, 'logs/err.log'),
      merge_logs: true,
      time: true,
    },

    {
      // MT3 multi-instrument transcription worker. Runs in its OWN venv
      // (~/mt3-venv, PyTorch-only mt3-infer, torch 2.4.1 + transformers 4.44.2
      // — versions are delicate, see deploy/mt3-setup.sh). Never touches the
      // main MusicNote deps. Optional — build with deploy/mt3-setup.sh.
      name: 'mt3-worker',
      script: path.join(process.env.HOME || '/home/ubuntu', 'mt3-venv/bin/python'),
      args: path.join(ROOT, 'backend/mt3_worker.py'),
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10,
      kill_timeout: 10000,
      // Below what the kernel will act at on an 11 GB box, so pm2 restarts the
      // worker between requests instead of the OOM killer taking it mid-response
      // (which the app sees as RemoteDisconnected and reports as a dead worker).
      // With MT3_BATCH_SEGMENTS capping the inference batch the peak is far
      // under this; the ceiling is here for the case where it is not.
      max_memory_restart: '6000M',
      env: {
        PYTHONUNBUFFERED: '1',
        // 'yourmt3' = best quality, multi-track (~7.5 GB peak, released when idle).
        // 'mr_mt3' = light (~0.7 GB) but much weaker (collapses to ~2 tracks).
        MT3_MODEL: 'yourmt3',
        MT3_PORT: '8732',
        MT3_THREADS: '2',
        MT3_IDLE_UNLOAD: '600',      // release the model after 10 min idle
        MT3_CHECKPOINT_DIR: path.join(process.env.HOME || '/home/ubuntu', 'mt3-ckpts'),
      },
      out_file: path.join(ROOT, 'logs/mt3.out.log'),
      error_file: path.join(ROOT, 'logs/mt3.err.log'),
      merge_logs: true,
      time: true,
    },

    {
      // bgutil PO-token provider — companion service for yt-dlp / YouTube.
      // Needs Node >= 22 (private copy under ~/.local, see deploy/bgutil-pot.sh).
      name: 'bgutil-pot',
      script: path.join(ROOT, 'deploy/bgutil-pot.sh'),
      interpreter: 'bash',
      autorestart: true,
      max_restarts: 10,
      max_memory_restart: '500M',
      env: { BGUTIL_POT_PORT: '4416' },
      out_file: path.join(ROOT, 'logs/bgutil.out.log'),
      error_file: path.join(ROOT, 'logs/bgutil.err.log'),
      merge_logs: true,
      time: true,
    },
  ],
};
