(function() {
  'use strict';

  class TranscodeSession {
    constructor(isVod) {
      this.isVod = isVod;
      this.sessionId = null;
      this.generation = 0;
      this.pending = Promise.resolve();
      this.closed = false;
    }

    enqueue(action) {
      const operation = this.pending.then(action, action);
      this.pending = operation;
      return operation;
    }

    async release() {
      if (!this.sessionId) return;
      const response = await fetch(
        `/transcode/${encodeURIComponent(this.sessionId)}?force=${!this.isVod}`,
        { method: 'DELETE', keepalive: this.closed }
      );
      if (!response.ok) throw new Error(`Transcode stop failed: ${response.status}`);
      this.sessionId = null;
    }

    start(url) {
      const generation = ++this.generation;
      return this.enqueue(async () => {
        await this.release();
        if (this.closed || generation !== this.generation) return null;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Transcode start failed: ${response.status}`);
        const data = await response.json();
        if (!data.session_id) throw new Error('Transcode response has no session ID');
        this.sessionId = data.session_id;
        if (this.closed || generation !== this.generation) {
          await this.release();
          return null;
        }
        return data;
      });
    }

    stop() {
      ++this.generation;
      return this.enqueue(() => this.release());
    }

    detach() {
      // The receiver now owns this session; page-close cleanup must not stop it.
      ++this.generation;
      this.sessionId = null;
    }

    reopen() {
      this.closed = false;
    }

    close() {
      this.closed = true;
      ++this.generation;
      if (!this.sessionId) return;
      const url = `/transcode/${encodeURIComponent(this.sessionId)}/stop?force=${!this.isVod}`;
      if (!navigator.sendBeacon(url, new Blob([], { type: 'application/json' }))) {
        fetch(url, { method: 'POST', keepalive: true }).then(response => {
          if (!response.ok) throw new Error(`Transcode stop failed: ${response.status}`);
        }).catch(error => console.warn('[TC] Page-close cleanup failed:', error));
      }
    }
  }

  class AdaptiveLivePlayback {
    constructor({ video, hls, sessionId, playlist, onPlaylist }) {
      this.video = video;
      this.hls = hls;
      this.sessionId = sessionId;
      this.onPlaylist = onPlaylist;
      this.requestedPlaylist = null;
      this.bytes = 0;
      this.downloadMs = 0;
      this.requiredBitrate = 0;
      this.lastDownloadAt = 0;
      this.closed = false;
      this.abort = new AbortController();
      this.timer = null;
      this.requestTimer = null;
      this.lastError = null;
      this.onFragment = (_event, data) => {
        if (data.frag?.type !== 'main') return;
        const stats = data.part?.stats || data.frag.stats;
        const elapsed = stats?.loading?.end - stats?.loading?.start;
        if (!(stats?.loaded > 0 && elapsed > 0)) return;
        this.bytes += stats.loaded;
        this.downloadMs += elapsed;
        const duration = data.part?.duration || data.frag.duration;
        this.requiredBitrate = duration > 0 ? stats.loaded * 8 / duration : 0;
        this.lastDownloadAt = performance.now();
      };
      this.onManifest = () => {
        try {
          this.selectPlaylist(playlist);
        } catch (error) {
          console.warn('[LIVE] Unable to select initial rendition:', error);
        }
      };
      this.onLevel = (_event, data) => {
        const level = this.hls.levels[data.level];
        if (level) this.onPlaylist(this.levelUrl(level).pathname);
      };
      if (hls) {
        hls.startLevel = 0;
        hls.autoLevelCapping = 0;
        hls.on(Hls.Events.FRAG_LOADED, this.onFragment);
        hls.on(Hls.Events.MANIFEST_PARSED, this.onManifest);
        hls.on(Hls.Events.LEVEL_SWITCHED, this.onLevel);
      }
      this.poll();
    }

    levelUrl(level) {
      const url = level.uri || (Array.isArray(level.url) ? level.url[0] : level.url);
      return new URL(url, window.location.href);
    }

    selectPlaylist(playlist) {
      if (!this.hls || !this.hls.levels.length) return;
      const url = new URL(playlist, window.location.href);
      const prefix = `/transcode/${this.sessionId}/`;
      if (url.origin !== window.location.origin ||
          !['low.m3u8', 'high.m3u8'].some(name => url.pathname === prefix + name)) {
        throw new Error('Server returned an invalid live rendition');
      }
      const index = this.hls.levels.findIndex(level => {
        const levelUrl = this.levelUrl(level);
        return levelUrl.origin === url.origin && levelUrl.pathname === url.pathname;
      });
      if (index < 0) throw new Error('Requested live rendition is missing from the master playlist');
      if (this.requestedPlaylist === url.href) return;
      // Hls.js aligns the two renditions using their shared program dates and
      // switches fragments in the existing media buffer, without reopening input.
      this.hls.autoLevelCapping = index;
      this.hls.nextLevel = index;
      this.requestedPlaylist = url.href;
    }

    sample() {
      const video = this.video;
      let buffer = 0;
      for (let i = 0; i < video.buffered.length; i++) {
        if (video.buffered.start(i) <= video.currentTime &&
            video.currentTime <= video.buffered.end(i)) {
          buffer = Math.max(buffer, video.buffered.end(i) - video.currentTime);
        }
      }
      const paused = video.paused || video.ended;
      const fresh = performance.now() - this.lastDownloadAt <= 10000;
      const observed = !paused && fresh && this.downloadMs > 0
        ? this.bytes * 8000 / this.downloadMs : 0;
      this.bytes = 0;
      this.downloadMs = 0;
      return {
        buffer_seconds: Math.min(86400, Math.max(0, buffer)),
        waiting: !paused && video.readyState < 3,
        observed_bitrate: Number.isFinite(observed) ? Math.min(1e12, observed) : 0,
        required_bitrate: !paused && fresh ? Math.min(1e12, this.requiredBitrate) : 0,
      };
    }

    async poll() {
      if (this.closed) return;
      this.abort = new AbortController();
      this.requestTimer = setTimeout(() => this.abort.abort(), 8000);
      try {
        const response = await fetch(
          `/transcode/${encodeURIComponent(this.sessionId)}/health`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this.sample()),
            signal: this.abort.signal,
          }
        );
        if (!response.ok) throw new Error(`Playback health failed: ${response.status}`);
        const feedback = await response.json();
        if (this.closed) return;
        if (feedback.playlist && !this.video.paused && !this.video.ended) {
          this.selectPlaylist(feedback.playlist);
        }
        this.lastError = null;
      } catch (error) {
        if (!this.closed && this.lastError !== error.message) {
          console.warn('[LIVE] Keeping current rendition:', error);
          this.lastError = error.message;
        }
      } finally {
        clearTimeout(this.requestTimer);
        this.requestTimer = null;
        if (!this.closed) this.timer = setTimeout(() => this.poll(), 2000);
      }
    }

    destroy() {
      this.closed = true;
      this.abort.abort();
      clearTimeout(this.timer);
      clearTimeout(this.requestTimer);
      if (this.hls) {
        this.hls.off(Hls.Events.FRAG_LOADED, this.onFragment);
        this.hls.off(Hls.Events.MANIFEST_PARSED, this.onManifest);
        this.hls.off(Hls.Events.LEVEL_SWITCHED, this.onLevel);
      }
    }
  }

  window.NetvPlayback = { TranscodeSession, AdaptiveLivePlayback };
})();
