// IPTV Player Module
// Requires: Hls.js, window.PLAYER_CONFIG

(function() {
  'use strict';

  const cfg = window.PLAYER_CONFIG;
  const video = document.getElementById('video');
  const loading = document.getElementById('loading');
  const error = document.getElementById('error');
  const ccBtn = document.getElementById('toggle-cc');
  const settingsMenu = document.getElementById('settings-menu');

  // State
  let transcodeSessionId = null;
  let currentHls = null;
  let isTranscoding = false;
  let ccEnabled = cfg.captionsEnabled;
  let subtitlePollTimerId = null;
  let transcodedDuration = 0;
  let totalDuration = 0;
  let seekInProgress = false;
  let seekOffset = 0;
  let currentSubtitles = null;
  let activeTrackStates = null;
  let progressPollTimerId = null;
  let lastSavedPosition = 0;
  let savePositionTimeout = null;
  let autoMutedByPolicy = false;
  const transcodeSession = new window.NetvPlayback.TranscodeSession(cfg.isVod && !cfg.catchup);
  let adaptiveSession = null;
  let adaptivePlayback = null;
  let transcodePlaylist = null;
  let programStart = cfg.programStart || 0;
  let programEnd = cfg.programEnd || 0;
  // Seconds into the archived stream still to skip once they're transcoded
  let pendingCatchupSeek = cfg.catchup ? cfg.catchupSeek || 0 : 0;
  let programRefreshInFlight = false;
  let programRetryAt = 0;
  let castController = null;
  // AirPlay needs Safari's native HLS player; hls.js disables remote playback.
  let nativeHls = false;
  let scrubPercent = null;
  let archiveNavigationInProgress = false;

  // ============================================================
  // Utilities
  // ============================================================

  function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  function parseTime(str) {
    str = str.trim();
    if (!str) return 0;
    const parts = str.split(':').map(Number);
    if (parts.some(isNaN)) return 0;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    const n = parts[0];
    const duration = timelineDuration();
    if (duration >= 3600 && n * 3600 <= duration) return n * 3600;
    if (n * 60 <= duration) return n * 60;
    return n;
  }

  function parseVttTime(str) {
    const parts = str.split(':');
    if (parts.length === 3) return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    if (parts.length === 2) return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    return 0;
  }

  function parseVttCues(vttText) {
    const cues = [];
    const lines = vttText.split('\n');
    let i = 0;
    while (i < lines.length && !lines[i].includes('-->')) i++;
    while (i < lines.length) {
      const line = lines[i];
      if (line.includes('-->')) {
        const [startStr, endStr] = line.split('-->').map(s => s.trim().split(' ')[0]);
        const start = parseVttTime(startStr);
        const end = parseVttTime(endStr);
        i++;
        const textLines = [];
        while (i < lines.length && lines[i].trim() !== '') {
          textLines.push(lines[i]);
          i++;
        }
        if (textLines.length > 0) cues.push({start, end, text: textLines.join('\n')});
      }
      i++;
    }
    return cues;
  }

  // ============================================================
  // UI Helpers
  // ============================================================

  function hideLoading() {
    loading.classList.add('hidden');
  }

  function showLoading() {
    loading.classList.remove('hidden');
  }

  function showError() {
    adaptivePlayback?.destroy();
    adaptivePlayback = null;
    hideLoading();
    error.classList.remove('hidden');
  }

  function updateTranscodeCheck() {
    const check = document.getElementById('tc-check');
    if (check) check.textContent = isTranscoding ? '✓' : '';
  }

  function updateCcButton() {
    ccBtn.classList.toggle('active', ccEnabled);
  }

  function updatePlayIcon() {
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');
    if (playIcon && pauseIcon) {
      playIcon.classList.toggle('hidden', !video.paused);
      pauseIcon.classList.toggle('hidden', video.paused);
    }
  }

  function updateMuteIcon() {
    const volIcon = document.getElementById('vol-icon');
    const mutedIcon = document.getElementById('muted-icon');
    if (volIcon && mutedIcon) {
      volIcon.classList.toggle('hidden', video.muted);
      mutedIcon.classList.toggle('hidden', !video.muted);
    }
  }

  function updateFullscreenIcon() {
    const fsEnter = document.getElementById('fs-enter');
    const fsExit = document.getElementById('fs-exit');
    const isFs = !!document.fullscreenElement;
    if (fsEnter && fsExit) {
      fsEnter.classList.toggle('hidden', isFs);
      fsExit.classList.toggle('hidden', !isFs);
    }
  }

  function disableCcButton() {
    ccBtn.disabled = true;
    ccBtn.classList.remove('active');
  }

  function enableCcButton() {
    ccBtn.disabled = false;
    updateCcButton();
  }

  // ============================================================
  // HLS Configuration
  // ============================================================

  // Custom cueHandler for CEA-608 caption positioning
  // See: https://github.com/video-dev/hls.js/issues/654
  const customCueHandler = {
    newCue(track, startTime, endTime, captionScreen) {
      const lines = [];
      for (let r = 0; r < 15; r++) {
        const row = captionScreen.rows[r];
        let text = '';
        for (let c = 0; c < 32; c++) {
          text += row.chars[c]?.uchar || ' ';
        }
        text = text.trim();
        if (text) lines.push(text);
      }
      if (lines.length === 0) return [];
      const cue = new VTTCue(startTime, endTime, lines.join('\n'));
      cue.line = -2;
      cue.align = 'center';
      track.addCue(cue);
      return [cue];
    }
  };

  function createHlsConfig(options = {}) {
    const base = {
      enableWorker: true,
      lowLatencyMode: false,
      enableCEA708Captions: true,
      subtitleDisplay: ccEnabled,
      cueHandler: customCueHandler,
      manifestLoadingRetryDelay: 1000,
      levelLoadingRetryDelay: 1000,
      fragLoadingRetryDelay: 1000,
    };
    if (options.forSeek) {
      return {
        ...base,
        liveSyncDurationCount: 0,
        startPosition: 0,
        manifestLoadingMaxRetry: 30,
        levelLoadingMaxRetry: 30,
        fragLoadingMaxRetry: 30,
      };
    }
    return {
      ...base,
      liveSyncDurationCount: options.isVod ? 0 : 3,
      startPosition: options.isVod ? 0 : -1,
    };
  }

  // ============================================================
  // Captions
  // ============================================================

  function applyCaptionStyles() {
    const s = cfg.ccStyle || {};
    const hexToRgba = (hex, opacity) => {
      if (hex === 'transparent') return 'transparent';
      const r = parseInt(hex.slice(1,3), 16);
      const g = parseInt(hex.slice(3,5), 16);
      const b = parseInt(hex.slice(5,7), 16);
      return `rgba(${r},${g},${b},${opacity})`;
    };
    const color = hexToRgba(s.cc_color || '#ffffff', 1);
    const shadow = s.cc_shadow || '0 0 4px black, 0 0 4px black';
    const bg = hexToRgba(s.cc_bg || '#000000', s.cc_bg_opacity || 0.75);
    const size = s.cc_size || '1em';
    const font = s.cc_font || 'inherit';
    let style = document.getElementById('cc-style');
    if (!style) {
      style = document.createElement('style');
      style.id = 'cc-style';
      document.head.appendChild(style);
    }
    const sizeMultiplier = parseFloat(size) || 1;
    const infoSize = (2.5 * sizeMultiplier) + 'vh';
    style.textContent = `
      video::cue {
        color: ${color} !important;
        text-shadow: ${shadow} !important;
        background-color: ${bg} !important;
        font-size: ${size} !important;
        font-family: ${font} !important;
      }
      #info-overlay { font-size: ${infoSize}; max-width: 50em; }
    `;
  }

  function getPreferredSubtitleTrack(tracks) {
    const prefLang = cfg.ccLang || '';
    if (!prefLang || tracks.length === 0) return 0;
    // Handle CC1-CC4 (CEA-608 channels)
    if (/^cc[1-4]$/i.test(prefLang)) {
      const ccNum = prefLang.toUpperCase();
      const idx = tracks.findIndex(t => (t.name || t.label || '').toUpperCase().includes(ccNum));
      if (idx >= 0) return idx;
      // Fallback: CC1 is usually index 0, CC2 is index 1, etc.
      const num = parseInt(prefLang.slice(2)) - 1;
      return num < tracks.length ? num : 0;
    }
    const langNames = {en: 'english', es: 'spanish', fr: 'french', de: 'german', it: 'italian', pt: 'portuguese', ja: 'japanese', ko: 'korean', zh: 'chinese'};
    let idx = tracks.findIndex(t => t.lang && t.lang.toLowerCase().startsWith(prefLang));
    if (idx >= 0) return idx;
    const prefName = langNames[prefLang];
    if (prefName) {
      idx = tracks.findIndex(t => (t.name || t.label) && (t.name || t.label).toLowerCase().includes(prefName));
      if (idx >= 0) return idx;
    }
    return 0;
  }

  function applyCaptionsSetting() {
    const tracks = Array.from(video.textTracks).filter(t =>
      (t.kind === 'subtitles' || t.kind === 'captions') && t.mode !== 'disabled');
    if (!ccEnabled) {
      tracks.forEach(t => t.mode = 'hidden');
      return;
    }
    if (tracks.length === 0) return;
    const prefLang = cfg.ccLang || '';
    const langNames = {en: 'english', es: 'spanish', fr: 'french', de: 'german', it: 'italian', pt: 'portuguese', ja: 'japanese', ko: 'korean', zh: 'chinese'};
    let preferredIdx = 0;
    if (prefLang) {
      const idx = tracks.findIndex(t => t.language && t.language.toLowerCase().startsWith(prefLang));
      if (idx >= 0) preferredIdx = idx;
      else {
        const prefName = langNames[prefLang];
        if (prefName) {
          const nameIdx = tracks.findIndex(t => t.label && t.label.toLowerCase().includes(prefName));
          if (nameIdx >= 0) preferredIdx = nameIdx;
        }
      }
    }
    tracks.forEach((t, i) => t.mode = i === preferredIdx ? 'showing' : 'hidden');
  }

  function startSubtitlePolling(subtitles, prefIdx) {
    if (subtitlePollTimerId) {
      clearInterval(subtitlePollTimerId);
      clearTimeout(subtitlePollTimerId);
      subtitlePollTimerId = null;
    }
    if (activeTrackStates && activeTrackStates.length === subtitles.length) {
      for (const ts of activeTrackStates) {
        if (ts.track.mode === 'disabled') ts.track.mode = 'hidden';
        const cues = ts.track.cues;
        if (cues) while (cues.length > 0) ts.track.removeCue(cues[0]);
        ts.addedCues.clear();
        ts.retryCount = 0;
      }
    } else {
      for (let i = 0; i < video.textTracks.length; i++) {
        video.textTracks[i].mode = 'disabled';
      }
      activeTrackStates = subtitles.map((sub) => ({
        url: sub.url,
        track: video.addTextTrack('subtitles', sub.label, sub.lang),
        addedCues: new Set(),
        retryCount: 0,
      }));
    }
    activeTrackStates.forEach((ts, i) => {
      ts.track.mode = (ccEnabled && i === prefIdx) ? 'showing' : 'hidden';
    });

    const poll = async () => {
      for (let i = 0; i < activeTrackStates.length; i++) {
        const ts = activeTrackStates[i];
        if (ts.retryCount > 120) continue;
        try {
          const resp = await fetch(ts.url + '?t=' + Date.now());
          if (!resp.ok) { ts.retryCount++; continue; }
          const vtt = await resp.text();
          const cues = parseVttCues(vtt);
          for (const cue of cues) {
            const key = `${cue.start}-${cue.end}`;
            if (!ts.addedCues.has(key)) {
              try {
                ts.track.addCue(new VTTCue(cue.start, cue.end, cue.text));
                ts.addedCues.add(key);
              } catch (e) {}
            }
          }
          ts.retryCount = 0;
        } catch (e) { ts.retryCount++; }
      }
    };

    let pollCount = 0;
    const doPoll = async () => {
      await poll();
      pollCount++;
      subtitlePollTimerId = pollCount < 20 ? setTimeout(doPoll, 500) : setInterval(poll, 5000);
    };
    doPoll();
  }

  // ============================================================
  // Position Tracking
  // ============================================================

  function savePosition() {
    if (castController?.active) return;
    const actualTime = video.currentTime + seekOffset;
    if (!cfg.isVod || cfg.catchup || actualTime < 5) return;
    if (video.currentTime < 1 && seekOffset > 0) return;
    if (Math.abs(actualTime - lastSavedPosition) < 5) return;
    lastSavedPosition = actualTime;
    const data = JSON.stringify({ url: cfg.rawUrl, position: actualTime, duration: totalDuration });
    if (document.visibilityState === 'hidden') {
      navigator.sendBeacon('/api/watch-position', new Blob([data], {type: 'application/json'}));
    } else {
      fetch('/api/watch-position', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: data });
    }
  }

  function restorePosition() {
    if (!cfg.isVod) return;
    const savedTime = cfg.serverResumePosition;
    if (!savedTime || savedTime <= 5) return;
    if (transcodedDuration < 30) return;
    const rangeStart = seekOffset;
    const rangeEnd = seekOffset + Math.max(0, transcodedDuration - 10);
    if (savedTime < rangeStart) return;
    const targetTime = Math.min(savedTime, rangeEnd);
    video.currentTime = targetTime - seekOffset;
  }

  function setupPositionTracking() {
    if (!cfg.isVod) return;
    video.addEventListener('timeupdate', () => {
      if (savePositionTimeout) return;
      savePositionTimeout = setTimeout(() => {
        savePosition();
        savePositionTimeout = null;
      }, 10000);
    });
    video.addEventListener('pause', savePosition);
    video.addEventListener('ended', savePosition);
    video.addEventListener('ended', () => {
      if (autoNextEnabled && cfg.nextEpisodeUrl) window.location.href = cfg.nextEpisodeUrl;
    });
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') savePosition();
    });
  }

  // ============================================================
  // Live DVR Resume
  // ============================================================

  // MSE reports seekable as [0, duration] for live streams, so the
  // retained window must come from the active playlist's first
  // fragment. Native HLS (Safari) reports the real window in seekable.
  function liveWindowStart() {
    if (currentHls) {
      const levels = currentHls.levels || [];
      const level = levels[currentHls.currentLevel] || levels[currentHls.loadLevel] || levels[0];
      const fragments = level && level.details && level.details.fragments;
      if (fragments && fragments.length && Number.isFinite(fragments[0].start)) {
        return fragments[0].start;
      }
      return null;
    }
    return video.seekable.length ? video.seekable.start(0) : null;
  }

  // Playback position of the live edge: what plays when fully caught up.
  function liveEdge() {
    if (currentHls) {
      return Number.isFinite(currentHls.liveSyncPosition) ? currentHls.liveSyncPosition : null;
    }
    return video.seekable.length ? video.seekable.end(video.seekable.length - 1) : null;
  }

  // The server keeps recording while paused, within the DVR retention
  // window (cfg.liveDvrMins). If the paused position has fallen out of
  // the window by resume time, continue at the oldest available
  // position instead of jumping to the live edge.
  function setupLiveDvrResume() {
    if (cfg.isVod || !cfg.liveDvrMins) return;

    let pausedDuringSession = false;
    video.addEventListener('pause', () => { pausedDuringSession = true; });
    video.addEventListener('play', () => {
      if (!pausedDuringSession) return;
      const start = liveWindowStart();
      if (start === null || video.currentTime >= start) return;
      video.currentTime = start + 0.1;
    });
  }

  // ============================================================
  // Live Program Timeline
  // ============================================================

  // Live progress spans the scheduled EPG program, not the DVR segment
  // window: elapsed and remaining come from the program's start and
  // stop times, so pausing or rewinding moves the position back through
  // the program instead of restarting a segment counter.
  function hasLiveProgram() {
    return !cfg.isVod && programEnd > programStart;
  }

  // Wall-clock time of the frame on screen. Live playback trails the
  // broadcast by however far the current position sits behind the live
  // edge, which is also how a paused picture keeps its place.
  function playbackWallClock() {
    const now = Date.now() / 1000;
    const edge = liveEdge();
    if (edge === null || !Number.isFinite(video.currentTime)) return now;
    return now - Math.max(0, edge - video.currentTime);
  }

  // Seek to a moment of the broadcast. Moments older than the DVR holds
  // come from the upstream archive when the channel keeps one.
  function seekToWallClock(target) {
    const edge = liveEdge();
    const start = liveWindowStart();
    const now = Date.now() / 1000;
    const oldestHeld = edge === null || start === null ? now : now - (edge - start);
    if (cfg.catchupDays > 0 && target < oldestHeld - 5 && target >= archiveStart()) {
      openArchiveAt(target);
      return;
    }
    if (edge === null) return;
    const oldest = start === null ? 0 : start + 0.1;
    const position = edge - (Date.now() / 1000 - target);
    video.currentTime = Math.min(Math.max(position, oldest), edge);
  }

  // ============================================================
  // Catchup (upstream archive)
  // ============================================================

  // Oldest moment the upstream archive still holds, with a minute of margin.
  function archiveStart() {
    return Date.now() / 1000 - cfg.catchupDays * 86400 + 60;
  }

  // Reopen the channel at a moment of the broadcast: the upstream serves
  // an archive from any minute, which is how catchup seeks outside what's
  // already transcoded (archives can't be byte-seeked).
  async function openArchiveAt(target) {
    if (archiveNavigationInProgress) return;
    archiveNavigationInProgress = true;
    video.pause();
    showLoading();
    try {
      // Release the upstream connection before the next page starts probing.
      await cleanupTranscode();
      const base = '/play/live/' + encodeURIComponent(cfg.streamId);
      window.location.href = target >= Date.now() / 1000 - 30
        ? base
        : base + '?start=' + Math.floor(Math.max(target, archiveStart()));
    } catch (e) {
      archiveNavigationInProgress = false;
      console.error('[CATCHUP] Could not release the previous stream:', e);
      showError();
    }
  }

  function hasCatchupProgram() {
    return cfg.catchup && programEnd > programStart;
  }

  // Broadcast time of the frame on screen.
  function catchupClock() {
    return cfg.catchupStart + seekOffset + (video.currentTime || 0);
  }

  function seekCatchup(target) {
    pendingCatchupSeek = 0;
    const loadedEnd = cfg.catchupStart + seekOffset + Math.max(0, transcodedDuration - 2);
    if (target >= cfg.catchupStart + seekOffset && target <= loadedEnd) {
      video.currentTime = target - cfg.catchupStart - seekOffset;
      return;
    }
    return openArchiveAt(target);
  }

  function skip(seconds) {
    if (hasLiveProgram()) seekToWallClock(playbackWallClock() + seconds);
    else if (hasCatchupProgram()) seekCatchup(catchupClock() + seconds);
    else video.currentTime += seconds;
  }

  function applyPendingCatchupSeek() {
    if (pendingCatchupSeek <= 0 || transcodedDuration < pendingCatchupSeek + 2) return;
    video.currentTime = pendingCatchupSeek - seekOffset;
    pendingCatchupSeek = 0;
  }

  function setupCatchup() {
    if (!cfg.catchup) return;
    document.getElementById('go-live-btn')?.addEventListener('click', e => {
      e.preventDefault();
      openArchiveAt(Date.now() / 1000);
    });
    // Carry on with whatever aired next, or rejoin live once caught up.
    video.addEventListener('ended', () => openArchiveAt(Math.max(programEnd, catchupClock())));
  }

  // The guide only supplies the program that was on at page load, so
  // pull the next one once its stop time passes; a gap in the listings
  // is retried at a slow cadence.
  async function refreshProgram() {
    const now = Date.now() / 1000;
    if (programRefreshInFlight || !cfg.streamId || now < programRetryAt) return;
    programRefreshInFlight = true;
    programRetryAt = now + 30;
    try {
      const resp = await fetch('/api/live/program/' + encodeURIComponent(cfg.streamId));
      if (!resp.ok) return;
      const data = await resp.json();
      if (!data.end || data.end <= Date.now() / 1000) return;
      programStart = data.start;
      programEnd = data.end;
      const title = document.getElementById('program-title');
      if (title) title.textContent = data.title ? ' \u2014 ' + data.title : '';
      const desc = document.getElementById('program-desc');
      if (desc) {
        desc.textContent = data.desc || '';
        desc.classList.toggle('hidden', !data.desc);
      }
      updateProgress();
    } catch (e) {
      // Keep the current listing and try again on a later tick.
    } finally {
      programRefreshInFlight = false;
    }
  }

  function setupLiveProgram() {
    if (cfg.isVod) return;
    const container = document.getElementById('progress-container');
    const startOver = document.getElementById('start-over-btn');
    startOver?.addEventListener('click', () => {
      openArchiveAt(programStart);
    });
    // Wall-clock driven, so the bar keeps up while paused or stalled.
    const tick = () => {
      // Channels the guide knows nothing about stay off the timeline.
      if (programEnd && Date.now() / 1000 >= programEnd) refreshProgram();
      container?.classList.toggle('hidden', !hasLiveProgram());
      if (hasLiveProgram()) updateProgress();
      // The upstream archive only reaches back catchupDays
      const archiveStart = Date.now() / 1000 - cfg.catchupDays * 86400;
      startOver?.classList.toggle('hidden', !hasLiveProgram() || programStart < archiveStart);
    };
    tick();
    setInterval(tick, 1000);
  }

  // ============================================================
  // Progress Bar
  // ============================================================

  function renderProgress(pct, currentText, rightText) {
    const played = document.getElementById('progress-played');
    const handle = document.getElementById('progress-handle');
    const timeCurrent = document.getElementById('time-current');
    const timeDuration = document.getElementById('time-duration');
    if (played) played.style.width = pct + '%';
    if (handle) handle.style.left = pct + '%';
    if (timeCurrent) timeCurrent.textContent = currentText;
    if (timeDuration) timeDuration.textContent = rightText;
    const slider = document.getElementById('progress-seek');
    if (slider) {
      if (scrubPercent === null) slider.value = pct;
      slider.setAttribute('aria-valuetext', currentText);
    }
  }

  function timelineDuration() {
    if (hasCatchupProgram() || hasLiveProgram()) return programEnd - programStart;
    const duration = totalDuration || video.duration || 0;
    return Number.isFinite(duration) ? duration : 0;
  }

  // Shade the part of the program the DVR can still seek to.
  function renderDvrWindow(length) {
    const buffered = document.getElementById('progress-buffered');
    if (!buffered) return;
    const edge = liveEdge();
    const start = liveWindowStart();
    const now = Date.now() / 1000;
    const live = Math.min(Math.max(now - programStart, 0), length);
    const oldest = edge === null || start === null
      ? live
      : Math.min(Math.max(now - (edge - start) - programStart, 0), length);
    buffered.style.left = (oldest / length) * 100 + '%';
    buffered.style.width = ((live - oldest) / length) * 100 + '%';
  }

  // Shade the part of an archived program that's transcoded and seekable.
  function renderArchiveWindow(length) {
    const buffered = document.getElementById('progress-buffered');
    if (!buffered) return;
    const clamp = (t) => Math.min(Math.max(t - programStart, 0), length);
    const from = clamp(cfg.catchupStart + seekOffset);
    const to = clamp(cfg.catchupStart + seekOffset + transcodedDuration);
    buffered.style.left = (from / length) * 100 + '%';
    buffered.style.width = ((to - from) / length) * 100 + '%';
  }

  function updateProgress() {
    if (scrubPercent !== null) {
      const length = timelineDuration();
      const elapsed = length * scrubPercent / 100;
      renderProgress(scrubPercent, formatTime(elapsed),
        hasCatchupProgram() || hasLiveProgram() ? '-' + formatTime(length - elapsed) : formatTime(length));
      return;
    }
    if (hasCatchupProgram()) {
      const length = programEnd - programStart;
      const elapsed = Math.min(Math.max(catchupClock() - programStart, 0), length);
      renderProgress((elapsed / length) * 100, formatTime(elapsed), '-' + formatTime(length - elapsed));
      renderArchiveWindow(length);
      return;
    }
    if (hasLiveProgram()) {
      const length = programEnd - programStart;
      const elapsed = Math.min(Math.max(playbackWallClock() - programStart, 0), length);
      renderProgress((elapsed / length) * 100, formatTime(elapsed), '-' + formatTime(length - elapsed));
      renderDvrWindow(length);
      return;
    }
    const duration = totalDuration || video.duration || 0;
    if (!duration) return;
    const currentTime = video.currentTime + seekOffset;
    renderProgress((currentTime / duration) * 100, formatTime(currentTime), formatTime(duration));
  }

  // ============================================================
  // Progress Polling
  // ============================================================

  function startProgressPolling() {
    if (progressPollTimerId) clearInterval(progressPollTimerId);
    const poll = async () => {
      if (!transcodeSessionId) return;
      try {
        const resp = await fetch('/transcode/progress/' + transcodeSessionId);
        if (resp.ok) {
          const data = await resp.json();
          transcodedDuration = data.duration || 0;
          applyPendingCatchupSeek();
        }
      } catch (e) {}
    };
    poll();
    progressPollTimerId = setInterval(poll, 2000);
  }

  function stopProgressPolling() {
    if (progressPollTimerId) {
      clearInterval(progressPollTimerId);
      progressPollTimerId = null;
    }
  }

  // ============================================================
  // Transcode Management
  // ============================================================

  async function cleanupTranscode() {
    if (document.pictureInPictureElement === video && !castController?.active) return;
    adaptivePlayback?.destroy();
    adaptivePlayback = null;
    adaptiveSession = null;
    transcodePlaylist = null;
    stopProgressPolling();
    if (subtitlePollTimerId) {
      clearInterval(subtitlePollTimerId);
      subtitlePollTimerId = null;
    }
    if (currentHls) {
      currentHls.destroy();
      currentHls = null;
    }
    transcodedDuration = 0;
    totalDuration = 0;
    seekInProgress = false;
    seekOffset = 0;
    currentSubtitles = null;
    activeTrackStates = null;
    document.getElementById('menu-jump')?.classList.add('hidden');
    document.getElementById('seek-container').classList.add('hidden');
    await transcodeSession.stop();
    transcodeSessionId = null;
  }

  function cleanupTranscodeSync() {
    if (document.pictureInPictureElement === video) return;
    adaptivePlayback?.destroy();
    adaptivePlayback = null;
    stopProgressPolling();
    if (subtitlePollTimerId) {
      clearInterval(subtitlePollTimerId);
      subtitlePollTimerId = null;
    }
    if (currentHls) {
      currentHls.destroy();
      currentHls = null;
    }
    transcodeSession.close();
    transcodeSessionId = null;
  }

  async function handleSeekToPosition(targetTime) {
    // Upstream archives can't be byte-seeked, so a server-side seek would
    // re-read the recording from its start; reopen the archive instead.
    if (cfg.catchup) {
      seekCatchup(cfg.catchupStart + targetTime);
      return true;
    }
    if (!transcodeSessionId || seekInProgress) return false;
    seekInProgress = true;
    showLoading();
    error.classList.add('hidden');
    video.pause();
    video.src = '';
    if (subtitlePollTimerId) {
      clearInterval(subtitlePollTimerId);
      clearTimeout(subtitlePollTimerId);
      subtitlePollTimerId = null;
    }
    if (currentHls) {
      currentHls.destroy();
      currentHls = null;
    }
    try {
      const resp = await fetch('/transcode/seek/' + transcodeSessionId + '?time=' + targetTime);
      if (!resp.ok) throw new Error('Seek failed: ' + resp.status);
      transcodedDuration = 0;
      seekOffset = targetTime;
      if (nativeHls) {
        video.disableRemotePlayback = false;
        video.src = '/transcode/' + transcodeSessionId + '/stream.m3u8';
        video.addEventListener('loadedmetadata', () => {
          hideLoading();
          seekInProgress = false;
          savePosition();
          video.play().catch(() => {});
        }, { once: true });
        return true;
      }
      const hls = new Hls(createHlsConfig({forSeek: true}));
      currentHls = hls;

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        hideLoading();
        error.classList.add('hidden');
        seekInProgress = false;
        savePosition();
        if (currentSubtitles && currentSubtitles.length > 0) {
          startSubtitlePolling(currentSubtitles, getPreferredSubtitleTrack(currentSubtitles));
        }
        video.play().catch(() => {});
      });

      let recoveryAttempts = 0;
      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          recoveryAttempts++;
          if (recoveryAttempts <= 3) {
            if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad();
            else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
            else { console.error('[SEEK] HLS error:', data); showError(); }
          } else {
            console.error('[SEEK] HLS error after retries:', data);
            showError();
          }
        }
      });

      hls.loadSource('/transcode/' + transcodeSessionId + '/stream.m3u8');
      hls.attachMedia(video);
      return true;
    } catch (e) {
      console.error('[SEEK] Error:', e);
      seekInProgress = false;
      showError();
      return false;
    }
  }

  async function startTranscode(onError) {
    showLoading();
    try {
      await cleanupTranscode();
      if (castController?.active) return;
      // Archived programs are finite and seekable, so they play like a movie
      const contentType = cfg.catchup ? 'movie' : cfg.streamType;
      let url = '/transcode/start?url=' + encodeURIComponent(cfg.rawUrl) + '&content_type=' + contentType;
      if (cfg.seriesId) url += '&series_id=' + cfg.seriesId;
      if (cfg.episodeId) url += '&episode_id=' + cfg.episodeId;
      if (cfg.seriesName) url += '&series_name=' + encodeURIComponent(cfg.seriesName);
      if (cfg.deinterlaceFallback !== undefined) url += '&deinterlace_fallback=' + (cfg.deinterlaceFallback ? '1' : '0');
      if (cfg.sourceId) url += '&source_id=' + encodeURIComponent(cfg.sourceId);
      if (!cfg.isVod) url += '&fast_start=true';
      const data = await transcodeSession.start(url);
      if (!data) return;
      transcodeSessionId = data.session_id;
      transcodePlaylist = data.playlist;
      const playlistPath = new URL(data.playlist, window.location.href).pathname;
      adaptiveSession = !cfg.isVod && playlistPath.endsWith('/low.m3u8') ? data : null;
      isTranscoding = true;
      updateTranscodeCheck();
      totalDuration = data.duration || 0;
      seekOffset = data.seek_offset || 0;
      transcodedDuration = data.transcoded_duration || 0;
      currentSubtitles = data.subtitles || null;
      if (cfg.isVod) {
        document.getElementById('menu-jump')?.classList.remove('hidden');
        document.getElementById('progress-container')?.classList.remove('hidden');
        const duration = timelineDuration();
        if (duration > 0) {
          document.getElementById('seek-duration').textContent = '/ ' + formatTime(duration);
          document.getElementById('time-duration').textContent = formatTime(duration);
        }
        enableCcButton();
      }
      playWithUrl(data.master_playlist || data.playlist, onError, data.subtitles);
    } catch (e) {
      console.error('[TC] Error:', e);
      if (onError) onError();
      else showError();
    }
  }

  // ============================================================
  // Playback
  // ============================================================

  function playWithUrl(url, onError, subtitles) {
    showLoading();
    adaptivePlayback?.destroy();
    adaptivePlayback = null;
    const useHls = !nativeHls && Hls.isSupported() && (
      url.includes('.m3u8') || url.includes('/live/') || url.includes('/transcode')
    );
    if (!useHls) {
      video.disableRemotePlayback = false;
      // Native HLS without download telemetry stays on the safe initial rendition.
      video.src = adaptiveSession ? adaptiveSession.playlist : url;
      if (adaptiveSession) startAdaptivePlayback(null);
      video.addEventListener('loadedmetadata', function() {
        hideLoading();
        error.classList.add('hidden');
        if (video.textTracks.length === 0) disableCcButton();
        applyCaptionsSetting();
        restorePosition();
        if (transcodeSessionId && !adaptiveSession) startProgressPolling();
        video.play().catch(() => { if (!video.muted) { autoMutedByPolicy = true; video.muted = true; } video.play(); });
      }, { once: true });
      video.addEventListener('error', function() {
        if (onError) onError();
        else showError();
      }, { once: true });
      return;
    }

    const hls = new Hls({
      ...createHlsConfig({isVod: cfg.isVod}),
      ...(adaptiveSession ? { liveSyncDuration: 12, liveSyncDurationCount: undefined } : {}),
    });
    currentHls = hls;
    let recoveryAttempts = 0;
    let hasLoaded = false;

    if (adaptiveSession) {
      // Older servers can supply low.m3u8 without a master; keep their session
      // alive but do not request upgrades that cannot be switched safely.
      startAdaptivePlayback(adaptiveSession.master_playlist ? hls : null);
    }

    // Timeout for initial load (Auto mode only)
    let loadTimeout = null;
    if (onError) {
      loadTimeout = setTimeout(() => {
        if (!hasLoaded) {
          console.log('[AUTO] Load timeout, triggering transcode');
          hls.destroy();
          currentHls = null;
          onError();
        }
      }, 10000);
    }

    // Check for missing audio (Auto mode only)
    if (onError) {
      let audioChecked = false;
      video.addEventListener('timeupdate', function checkAudio() {
        if (audioChecked || video.currentTime < 1) return;
        audioChecked = true;
        video.removeEventListener('timeupdate', checkAudio);
        let hasAudio = false;
        if (typeof video.webkitAudioDecodedByteCount !== 'undefined') {
          hasAudio = video.webkitAudioDecodedByteCount > 0;
        } else if (typeof video.mozHasAudio !== 'undefined') {
          hasAudio = video.mozHasAudio;
        } else if (video.audioTracks && video.audioTracks.length > 0) {
          hasAudio = true;
        }
        console.log('[AUTO] Audio check: hasAudio=' + hasAudio + ', webkitAudioDecodedByteCount=' + video.webkitAudioDecodedByteCount);
        if (!hasAudio) {
          console.log('[AUTO] No audio detected, triggering transcode');
          hls.destroy();
          currentHls = null;
          onError();
        }
      });
    }

    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (loadTimeout) clearTimeout(loadTimeout);
      hideLoading();
      error.classList.add('hidden');
      hasLoaded = true;
      recoveryAttempts = 0;
      if (subtitles && subtitles.length > 0) {
        startSubtitlePolling(subtitles, getPreferredSubtitleTrack(subtitles));
      }
      if (cfg.captionsEnabled && hls.subtitleTracks.length > 0) {
        hls.subtitleTrack = getPreferredSubtitleTrack(hls.subtitleTracks);
      }
      if (transcodeSessionId && !adaptiveSession) startProgressPolling();
      restorePosition();
      video.play().catch(() => { if (!video.muted) { autoMutedByPolicy = true; video.muted = true; } video.play(); });
      setTimeout(() => {
        if (hls.subtitleTracks.length === 0 && video.textTracks.length === 0) disableCcButton();
      }, 1000);
    });

    hls.on(Hls.Events.SUBTITLE_TRACKS_UPDATED, () => {
      if (hls.subtitleTracks.length > 0 && ccBtn.disabled) {
        enableCcButton();
        if (cfg.captionsEnabled) hls.subtitleTrack = getPreferredSubtitleTrack(hls.subtitleTracks);
      } else if (hls.subtitleTracks.length === 0 && !ccBtn.disabled) {
        disableCcButton();
      }
    });

    video.textTracks.addEventListener('addtrack', (e) => {
      if (e.track.kind === 'captions' || e.track.kind === 'subtitles') {
        applyCaptionsSetting();
        if (ccBtn.disabled) enableCcButton();
      }
    });

    hls.on(Hls.Events.ERROR, (event, data) => {
      if (data.fatal) {
        recoveryAttempts++;
        if (recoveryAttempts <= 3) {
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad();
          else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
          else {
            hls.destroy();
            currentHls = null;
            if (!hasLoaded && onError) onError();
            else showError();
          }
        } else {
          hls.destroy();
          currentHls = null;
          if (!hasLoaded && onError) onError();
          else showError();
        }
      }
    });
    hls.loadSource(url);
    hls.attachMedia(video);
  }

  function startAdaptivePlayback(hls) {
    adaptivePlayback = new window.NetvPlayback.AdaptiveLivePlayback({
      video,
      hls,
      sessionId: transcodeSessionId,
      playlist: adaptiveSession.playlist,
      onPlaylist: playlist => { transcodePlaylist = playlist; },
    });
  }

  // ============================================================
  // Controls
  // ============================================================

  function setupKeyboardControls() {
    document.addEventListener('keydown', (e) => {
      if (castController?.active || document.getElementById('cast-dialog')?.open) return;
      // In seek input: allow player hotkeys, block other non-time chars
      if (e.target.id === 'seek-input') {
        const passthrough = ['j', 'm', 'f', ' ', 'k', 'c', 'i', 'Escape'];
        if (!passthrough.includes(e.key) && !/^[0-9:]$/.test(e.key) &&
            !['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'Tab', 'Enter'].includes(e.key)) {
          e.preventDefault();
          return;
        }
        // Let passthrough keys fall through to main handler below
        if (!passthrough.includes(e.key)) return;
      }
      // Skip other input fields entirely
      else if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') e.target.blur();
        return;
      }
      switch(e.key) {
        case ' ':
        case 'k':
          if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') return;
          e.preventDefault();
          video.paused ? video.play() : video.pause();
          break;
        case 'ArrowLeft': e.preventDefault(); skip(-10); break;
        case 'ArrowRight': e.preventDefault(); skip(10); break;
        case 'ArrowUp': e.preventDefault(); video.volume = Math.min(1, video.volume + 0.1); break;
        case 'ArrowDown': e.preventDefault(); video.volume = Math.max(0, video.volume - 0.1); break;
        case 'f':
          e.preventDefault();
          suppressShowControls = true;
          setTimeout(() => suppressShowControls = false, 150);
          if (document.fullscreenElement) document.exitFullscreen();
          else document.getElementById('player-container').requestFullscreen();
          break;
        case 'm': video.muted = !video.muted; updateMuteIcon(); break;
        case 't': document.getElementById('menu-transcode')?.click(); break;
        case 'c': ccBtn.click(); break;
        case 'i': document.getElementById('info-btn')?.click(); break;
        case 'a': document.getElementById('cast-btn')?.click(); break;
        case 'x': document.getElementById('menu-restart')?.click(); break;
        case 'j':
          e.preventDefault();
          if (cfg.isVod) document.getElementById('jump-btn')?.click();
          break;
        case 'n': document.getElementById('autonext-btn')?.click(); break;
        case 'p': document.getElementById('pip-btn')?.click(); break;
        case 'h':
          const container = document.getElementById('player-container');
          if (container.classList.contains('controls-visible')) {
            container.classList.remove('controls-visible');
            clearTimeout(activityTimeoutId);
            activityTimeoutId = null;
          } else {
            container.classList.add('controls-visible');
            clearTimeout(activityTimeoutId);
            activityTimeoutId = setTimeout(() => {
              if (!settingsMenu.classList.contains('open')) {
                container.classList.remove('controls-visible');
                activityTimeoutId = null;
              }
            }, 3000);
          }
          break;
        case 'Escape': {
          const seekContainer = document.getElementById('seek-container');
          const infoOverlay = document.getElementById('info-overlay');
          if (seekContainer && !seekContainer.classList.contains('hidden')) {
            seekContainer.classList.add('hidden');
          } else if (infoOverlay?.classList.contains('pinned')) {
            infoOverlay.classList.remove('pinned');
            document.getElementById('info-btn')?.classList.remove('active');
            saveSettings({ infoPinned: false });
          } else if (settingsMenu.classList.contains('open')) {
            settingsMenu.classList.remove('open');
          } else if (!document.fullscreenElement) {
            if (history.length > 1) history.back();
            else window.location.href = '/guide';
          }
          break;
        }
      }
    });
  }

  let activityTimeoutId = null;
  let autoNextEnabled = true;
  let suppressShowControls = false;

  // Persistent settings
  const STORAGE_KEY = 'playerSettings';
  function loadSettings() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch { return {}; }
  }
  function saveSettings(updates) {
    const settings = { ...loadSettings(), ...updates };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }

  function setupActivityTracking() {
    const container = document.getElementById('player-container');
    const HIDE_DELAY = 3000;

    function showControls() {
      if (suppressShowControls) return;
      container.classList.add('controls-visible');
      clearTimeout(activityTimeoutId);
      activityTimeoutId = setTimeout(hideControls, HIDE_DELAY);
    }

    function hideControls() {
      if (scrubPercent !== null) return;
      if (settingsMenu.classList.contains('open')) return;
      if (document.getElementById('seek-container')?.classList.contains('hidden') === false) return;
      container.classList.remove('controls-visible');
      activityTimeoutId = null;
    }

    container.addEventListener('mousemove', showControls);
    container.addEventListener('mouseenter', showControls);
    container.addEventListener('click', showControls);
    container.addEventListener('mouseleave', () => {
      clearTimeout(activityTimeoutId);
      activityTimeoutId = null;
      hideControls();
    });
    showControls();
  }

  function setupButtonHandlers() {
    const seekContainer = document.getElementById('seek-container');
    const seekInput = document.getElementById('seek-input');

    // Play/Pause button
    document.getElementById('play-btn')?.addEventListener('click', () => {
      video.paused ? video.play() : video.pause();
    });

    // Click video to toggle play/pause and show controls
    video.addEventListener('click', (e) => {
      if (e.target !== video) return;
      video.paused ? video.play() : video.pause();
      const container = document.getElementById('player-container');
      container.classList.add('controls-visible');
      clearTimeout(activityTimeoutId);
      activityTimeoutId = setTimeout(() => {
        if (scrubPercent === null && !settingsMenu.classList.contains('open') && seekContainer.classList.contains('hidden')) {
          container.classList.remove('controls-visible');
          activityTimeoutId = null;
        }
      }, 3000);
    });

    // Mute button
    document.getElementById('mute-btn')?.addEventListener('click', () => {
      video.muted = !video.muted;
      updateMuteIcon();
      saveSettings({ muted: video.muted });
    });

    // Volume slider
    const volSlider = document.getElementById('volume-slider');
    volSlider?.addEventListener('input', () => {
      video.volume = parseFloat(volSlider.value);
      video.muted = false;
      updateMuteIcon();
      saveSettings({ volume: video.volume, muted: false });
    });
    video.addEventListener('volumechange', () => {
      if (volSlider) volSlider.value = video.muted ? 0 : video.volume;
    });

    // Jump button
    document.getElementById('jump-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      seekContainer.classList.toggle('hidden');
      if (!seekContainer.classList.contains('hidden')) {
        // Show controls and prevent auto-hide while jump input is active
        document.getElementById('player-container').classList.add('controls-visible');
        clearTimeout(activityTimeoutId);
        activityTimeoutId = null;
        seekInput.value = '';
        seekInput.focus();
      }
    });

    // Auto-next button
    const autoNextBtn = document.getElementById('autonext-btn');
    autoNextBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      autoNextEnabled = !autoNextEnabled;
      autoNextBtn.classList.toggle('active', autoNextEnabled);
    });

    // Info button
    const infoOverlay = document.getElementById('info-overlay');
    const infoBtn = document.getElementById('info-btn');
    infoBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      const wasPinned = infoOverlay.classList.contains('pinned');
      infoOverlay.classList.toggle('pinned');
      infoBtn.classList.toggle('active', !wasPinned);
      saveSettings({ infoPinned: !wasPinned });
      if (wasPinned && !activityTimeoutId) {
        document.getElementById('player-container').classList.remove('controls-visible');
      }
    });

    // CC button
    ccBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      ccEnabled = !ccEnabled;
      if (activeTrackStates && activeTrackStates.length > 0) {
        const prefIdx = getPreferredSubtitleTrack(activeTrackStates.map(ts => ({lang: ts.track.language, label: ts.track.label})));
        activeTrackStates.forEach((ts, i) => ts.track.mode = (ccEnabled && i === prefIdx) ? 'showing' : 'hidden');
      } else {
        const tracks = Array.from(video.textTracks).filter(t => (t.kind === 'subtitles' || t.kind === 'captions') && t.mode !== 'disabled');
        const prefIdx = getPreferredSubtitleTrack(tracks.map(t => ({lang: t.language, label: t.label})));
        tracks.forEach((t, i) => t.mode = (ccEnabled && i === prefIdx) ? 'showing' : 'hidden');
      }
      if (currentHls) {
        currentHls.subtitleDisplay = ccEnabled;
        if (ccEnabled && currentHls.subtitleTracks?.length > 0) {
          currentHls.subtitleTrack = getPreferredSubtitleTrack(currentHls.subtitleTracks);
        }
      }
      updateCcButton();
      saveSettings({ ccEnabled });
    });

    // PiP button
    const pipBtn = document.getElementById('pip-btn');
    if (pipBtn && document.pictureInPictureEnabled) {
      pipBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          if (document.pictureInPictureElement) {
            await document.exitPictureInPicture();
          } else {
            await video.requestPictureInPicture();
          }
        } catch (err) {
          console.error('[PiP] Error:', err);
        }
      });
    } else if (pipBtn) {
      pipBtn.style.display = 'none';
    }

    // Settings menu toggle
    document.getElementById('settings-btn')?.addEventListener('click', () => {
      settingsMenu.classList.toggle('open');
    });

    // Close settings menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#settings-btn') && !e.target.closest('#settings-menu')) {
        settingsMenu.classList.remove('open');
      }
    });

    // CC Track selection
    let selectedCcTrackIdx = 0;
    const ccTracksMenuItem = document.getElementById('menu-cc-tracks');

    function getCcTracks() {
      const tracks = [];
      if (activeTrackStates?.length > 0) {
        activeTrackStates.forEach((ts, i) => tracks.push({ idx: i, label: ts.track.label || `Track ${i + 1}`, lang: ts.track.language }));
      } else if (currentHls?.subtitleTracks?.length > 0) {
        currentHls.subtitleTracks.forEach((t, i) => tracks.push({ idx: i, label: t.name || `Track ${i + 1}`, lang: t.lang }));
      } else {
        Array.from(video.textTracks).filter(t => t.kind === 'subtitles' || t.kind === 'captions')
          .forEach((t, i) => tracks.push({ idx: i, label: t.label || `Track ${i + 1}`, lang: t.language }));
      }
      return tracks;
    }

    function selectCcTrack(idx) {
      ccEnabled = true;
      updateCcButton();
      if (activeTrackStates?.length > 0) {
        activeTrackStates.forEach((ts, i) => ts.track.mode = i === idx ? 'showing' : 'hidden');
      } else if (currentHls?.subtitleTracks?.length > 0) {
        currentHls.subtitleTrack = idx;
        currentHls.subtitleDisplay = true;
        // iOS: also set TextTrack.mode directly (HLS.js property change doesn't always trigger render)
        const tracks = Array.from(video.textTracks).filter(
          t => t.kind === 'subtitles' || t.kind === 'captions'
        );
        tracks.forEach((t, i) => t.mode = i === idx ? 'showing' : 'hidden');
      } else {
        const tracks = Array.from(video.textTracks).filter(t => t.kind === 'subtitles' || t.kind === 'captions');
        tracks.forEach((t, i) => t.mode = i === idx ? 'showing' : 'hidden');
      }
    }

    function updateCcTracksLabel() {
      if (!ccTracksMenuItem) return;
      const tracks = getCcTracks();
      const label = tracks.length > 0 && tracks[selectedCcTrackIdx]
        ? `CC: ${tracks[selectedCcTrackIdx].label}`
        : 'CC Track';
      ccTracksMenuItem.innerHTML = `<span class="settings-check"></span>${label}`;
    }

    ccTracksMenuItem?.addEventListener('click', () => {
      const tracks = getCcTracks();
      if (tracks.length === 0) return;
      selectedCcTrackIdx = (selectedCcTrackIdx + 1) % tracks.length;
      selectCcTrack(selectedCcTrackIdx);
      updateCcTracksLabel();
    });

    // Settings menu items
    document.getElementById('menu-transcode')?.addEventListener('click', async () => {
      settingsMenu.classList.remove('open');
      video.pause();
      video.src = '';
      error.classList.add('hidden');
      if (isTranscoding) {
        try {
          await cleanupTranscode();
          isTranscoding = false;
          updateTranscodeCheck();
          playWithUrl(cfg.rawUrl);
        } catch (e) {
          console.error('[TC] Unable to stop transcode:', e);
          showError();
        }
      } else {
        await startTranscode();
      }
    });

    document.getElementById('menu-restart')?.addEventListener('click', async () => {
      settingsMenu.classList.remove('open');
      video.pause();
      video.src = '';
      error.classList.add('hidden');
      try {
        await fetch('/transcode-clear?url=' + encodeURIComponent(cfg.rawUrl), {method: 'DELETE'});
        await cleanupTranscode();
        isTranscoding = false;
        await startTranscode();
      } catch (e) {
        console.error('[X] Error:', e);
        showError();
      }
    });

    document.getElementById('menu-jump')?.addEventListener('click', () => {
      settingsMenu.classList.remove('open');
      seekContainer.classList.toggle('hidden');
      if (!seekContainer.classList.contains('hidden')) {
        // Show controls and prevent auto-hide while jump input is active
        document.getElementById('player-container').classList.add('controls-visible');
        clearTimeout(activityTimeoutId);
        activityTimeoutId = null;
        seekInput.value = '';
        seekInput.focus();
      }
    });

    document.getElementById('menu-url')?.addEventListener('click', () => {
      settingsMenu.classList.remove('open');
      if (navigator.clipboard) {
        navigator.clipboard.writeText(cfg.rawUrl).catch(fallback);
      } else {
        fallback();
      }
      function fallback() {
        const ta = document.createElement('textarea');
        ta.value = cfg.rawUrl;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
    });

    document.getElementById('menu-external')?.addEventListener('click', () => {
      settingsMenu.classList.remove('open');
      video.pause();
      video.src = '';
      window.location.href = '/playlist.xspf?url=' + encodeURIComponent(cfg.rawUrl);
    });

    // Fullscreen button
    document.getElementById('fullscreen-btn')?.addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.getElementById('player-container').requestFullscreen();
    });

    // Fullscreen change listener
    document.addEventListener('fullscreenchange', updateFullscreenIcon);

    // Video state listeners
    video.addEventListener('play', updatePlayIcon);
    video.addEventListener('pause', updatePlayIcon);
    video.addEventListener('volumechange', updateMuteIcon);

    // Prevent video element's native spacebar handling
    video.addEventListener('keydown', (e) => {
      if (e.key === ' ') e.preventDefault();
    });

    // Mousewheel volume control
    document.getElementById('player-container')?.addEventListener('wheel', (e) => {
      e.preventDefault();
      video.volume = Math.max(0, Math.min(1, video.volume + (e.deltaY < 0 ? 0.05 : -0.05)));
      saveSettings({ volume: video.volume });
    }, { passive: false });

    // Progress bar
    const progressBar = document.getElementById('progress-bar');
    const progressSeek = document.getElementById('progress-seek');

    video.addEventListener('timeupdate', updateProgress);
    video.addEventListener('loadedmetadata', () => {
      if (cfg.isVod && !transcodeSessionId) {
        document.getElementById('progress-container')?.classList.remove('hidden');
        document.getElementById('menu-jump')?.classList.remove('hidden');
      }
      updateProgress();
    });

    async function seekToProgress(pct) {
      pct = Math.max(0, Math.min(pct, 1));
      // Live clicks land on a moment of the broadcast, not a file offset.
      if (hasLiveProgram()) {
        seekToWallClock(programStart + pct * (programEnd - programStart));
        updateProgress();
        return;
      }
      if (hasCatchupProgram()) {
        await seekCatchup(programStart + pct * (programEnd - programStart));
        updateProgress();
        return;
      }
      const duration = timelineDuration();
      if (!duration) return;
      const targetTime = pct * duration;
      if (transcodeSessionId) {
        const actualTranscodedEnd = seekOffset + transcodedDuration;
        if (targetTime >= seekOffset && targetTime <= actualTranscodedEnd + 10) {
          video.currentTime = targetTime - seekOffset;
        } else {
          await handleSeekToPosition(targetTime);
        }
      } else {
        video.currentTime = targetTime;
      }
    }

    progressBar?.addEventListener('click', async (e) => {
      const rect = progressBar.getBoundingClientRect();
      if (rect.width > 0) await seekToProgress((e.clientX - rect.left) / rect.width);
    });
    // The native range handles mouse, touch and keyboard dragging. Preview
    // while moving; open at most one archive request when the seek is committed.
    progressSeek?.addEventListener('click', e => e.stopPropagation());
    progressSeek?.addEventListener('input', () => {
      scrubPercent = Number(progressSeek.value);
      document.getElementById('player-container').classList.add('controls-visible');
      updateProgress();
    });
    progressSeek?.addEventListener('change', async () => {
      const pct = Number(progressSeek.value) / 100;
      scrubPercent = null;
      await seekToProgress(pct);
      updateProgress();
    });
    const cancelScrub = () => {
      scrubPercent = null;
      updateProgress();
    };
    progressSeek?.addEventListener('pointercancel', cancelScrub);
    progressSeek?.addEventListener('blur', cancelScrub);

    // Seek input handler - filter chars here (must be at input level to block typing)
    seekInput?.addEventListener('keydown', async function(e) {
      // Only allow: digits, colon, navigation keys, Enter
      // Hotkeys and other chars: preventDefault (hotkeys will still bubble to global handler)
      const typeable = /^[0-9:]$/.test(e.key) ||
        ['Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'Tab'].includes(e.key);
      if (!typeable) e.preventDefault();
      if (e.key !== 'Enter') return;
      e.preventDefault();
      const targetTime = parseTime(seekInput.value);
      if (targetTime < 0 || targetTime > timelineDuration()) {
        seekInput.classList.add('ring-2', 'ring-red-500');
        setTimeout(() => seekInput.classList.remove('ring-2', 'ring-red-500'), 500);
        return;
      }
      seekContainer.classList.add('hidden');
      if (cfg.catchup) {
        await seekCatchup(programStart + targetTime);
        updateProgress();
        return;
      }
      const actualTranscodedEnd = seekOffset + transcodedDuration;
      if (targetTime >= seekOffset && targetTime <= actualTranscodedEnd + 10) {
        video.currentTime = targetTime - seekOffset;
        return;
      }
      await handleSeekToPosition(targetTime);
    });
  }

  // ============================================================
  // Cast (Chromecast)
  // ============================================================

  function setupCast() {
    castController = window.NetvCast.setup({
      serverAddress: cfg.castHost,
      getMedia: async () => {
        const position = video.currentTime + seekOffset;
        if (!transcodeSessionId) {
          await startTranscode();
          if (!transcodeSessionId) throw new Error('Could not prepare an HLS stream for the TV.');
          if (cfg.isVod && position > 0) {
            if (!await handleSeekToPosition(position)) throw new Error('Could not prepare the current movie position.');
          }
        }
        return {
          session_id: transcodeSessionId, title: cfg.mediaTitle,
          current_time: Math.max(0, position - seekOffset),
        };
      },
      onStarted: async status => {
        video.pause();
        if (document.pictureInPictureElement === video) {
          await document.exitPictureInPicture().catch(e => console.warn('[CAST] Could not close picture-in-picture:', e));
        }
        if (status.session_id === transcodeSessionId) transcodeSession.detach();
        await cleanupTranscode();
        video.removeAttribute('src');
        video.load();
        hideLoading();
        error.classList.add('hidden');
      },
    });
  }

  // ============================================================
  // AirPlay (Safari)
  // ============================================================

  function setupAirPlay() {
    const btn = document.getElementById('airplay-btn');
    const status = document.getElementById('airplay-status');
    if (!btn || !window.WebKitPlaybackTargetAvailabilityEvent || !video.webkitShowPlaybackTargetPicker) return;
    btn.classList.remove('hidden');
    let statusTimer = null;
    let preparing = false;

    function notify(text) {
      status.textContent = text;
      status.classList.remove('hidden');
      clearTimeout(statusTimer);
      statusTimer = setTimeout(() => status.classList.add('hidden'), 8000);
    }

    video.addEventListener('webkitcurrentplaybacktargetiswirelesschanged', () => {
      const wireless = !!video.webkitCurrentPlaybackTargetIsWireless;
      btn.classList.toggle('active', wireless);
      notify(wireless ? 'Playing on AirPlay.' : 'AirPlay ended. Playing here.');
    });

    btn.addEventListener('click', async e => {
      e.stopPropagation();
      if (preparing) return;
      if (castController?.active) return notify('Stop casting to Chromecast before using AirPlay.');
      if (['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)) {
        return notify('Open neTV using its LAN address (not localhost) so the TV can reach it.');
      }
      if (nativeHls && transcodeSessionId) return video.webkitShowPlaybackTargetPicker();
      if (transcodeSessionId) {
        // The picker needs this click's user gesture, so switch players synchronously.
        const position = cfg.isVod ? video.currentTime : 0;
        nativeHls = true;
        if (currentHls) { currentHls.destroy(); currentHls = null; }
        playWithUrl(transcodePlaylist, null, currentSubtitles);
        if (position > 0) {
          video.addEventListener('loadedmetadata', () => { video.currentTime = position; }, { once: true });
        }
        return video.webkitShowPlaybackTargetPicker();
      }
      // Direct playback must first move to a neTV HLS stream the TV can fetch.
      preparing = true;
      btn.disabled = true;
      notify('Preparing the stream for AirPlay...');
      try {
        const position = video.currentTime + seekOffset;
        nativeHls = true;
        await startTranscode();
        if (!transcodeSessionId) throw new Error('no transcode session');
        if (cfg.isVod && position > 0 && !await handleSeekToPosition(position)) {
          throw new Error('could not restore position');
        }
        notify('AirPlay is ready. Tap AirPlay again to choose a TV.');
      } catch (err) {
        console.warn('[AIRPLAY]', err);
        notify('Could not prepare the stream for AirPlay.');
      } finally {
        preparing = false;
        btn.disabled = false;
      }
    });
  }

  // ============================================================
  // Initialization
  // ============================================================

  function init() {
    // Restore persistent settings
    const settings = loadSettings();
    if (settings.volume !== undefined) video.volume = settings.volume;
    if (settings.muted !== undefined) video.muted = settings.muted;
    if (settings.ccEnabled !== undefined) ccEnabled = settings.ccEnabled;

    applyCaptionStyles();
    setupPositionTracking();
    setupLiveDvrResume();
    setupLiveProgram();
    setupCatchup();
    setupKeyboardControls();
    setupButtonHandlers();
    setupActivityTracking();
    setupCast();
    setupAirPlay();
    updateTranscodeCheck();
    updateCcButton();
    updateMuteIcon();
    document.getElementById('volume-slider').value = video.muted ? 0 : video.volume;

    // Restore volume on first user interaction if auto-muted by browser policy
    function restoreVolumeOnInteraction() {
      if (autoMutedByPolicy) {
        video.muted = false;
        updateMuteIcon();
        autoMutedByPolicy = false;
      }
    }
    document.addEventListener('click', restoreVolumeOnInteraction, { once: true });
    document.addEventListener('keydown', restoreVolumeOnInteraction, { once: true });

    // Restore info pinned state
    if (settings.infoPinned) {
      document.getElementById('info-overlay')?.classList.add('pinned');
      document.getElementById('info-btn')?.classList.add('active');
    }


    window.addEventListener('beforeunload', cleanupTranscodeSync);
    window.addEventListener('pagehide', cleanupTranscodeSync);
    window.addEventListener('pageshow', event => {
      if (event.persisted && transcodeSession.closed && !castController?.active) {
        transcodeSession.reopen();
        if (isTranscoding) startTranscode();
        else playWithUrl(cfg.rawUrl);
      }
    });

    // Do not open a second provider connection when returning to an active cast.
    castController.ready.then(() => {
      if (castController.active) return;
      if (cfg.transcodeMode === 'always') {
        startTranscode();
      } else if (cfg.transcodeMode === 'never') {
        playWithUrl(cfg.rawUrl);
      } else {
        playWithUrl(cfg.rawUrl, () => {
          error.classList.add('hidden');
          startTranscode();
        });
      }
    });
  }

  init();
})();
