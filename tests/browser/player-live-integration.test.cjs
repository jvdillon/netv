const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

const flush = () => new Promise((resolve) => setImmediate(resolve));

// Percentages ride on Date.now(), so compare them to the millisecond.
const assertPercent = (actual, expected) =>
  assert.ok(
    Math.abs(parseFloat(actual) - expected) < 0.01,
    `${actual} is not about ${expected}%`,
  );

function element() {
  const listeners = new Map();
  const classes = new Set(["hidden"]);
  const attributes = new Map();
  return {
    style: {},
    disabled: false,
    value: "",
    textContent: "",
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, enabled) => {
        if (enabled ?? !classes.has(name)) classes.add(name);
        else classes.delete(name);
      },
    },
    addEventListener: (name, fn) => {
      const handlers = listeners.get(name) || [];
      handlers.push(fn);
      listeners.set(name, handlers);
    },
    removeEventListener: (name, fn) => {
      listeners.set(
        name,
        (listeners.get(name) || []).filter((handler) => handler !== fn),
      );
    },
    async emit(name, value = {}) {
      for (const handler of listeners.get(name) || []) await handler(value);
    },
    setAttribute: (name, value) => attributes.set(name, String(value)),
    getAttribute: (name) => attributes.get(name),
    removeAttribute() {},
    appendChild() {},
    replaceChildren() {},
    showModal() {
      this.open = true;
    },
    close() {
      this.open = false;
    },
    getBoundingClientRect: () => ({ left: 0, width: 100, top: 0, height: 4 }),
  };
}

async function player({
  isVod = false,
  native = false,
  direct = false,
  auto = false,
  noManifest = false,
  programStart = 0,
  programEnd = 0,
  nextProgram = null,
  castFailure = false,
  initialCast = null,
  seekOffset = 0,
  statusReady = null,
  airplay = false,
  browserCast = false,
  catchup = false,
  catchupDays = catchup ? 2 : 0,
  catchupStart = programStart,
  catchupSeek = 0,
  transcodedDuration = 0,
  stopReady = null,
  stopFailure = false,
  startReady = null,
  startGeneration = 0,
  playbackBufferSeconds = 12,
  captionsEnabled = false,
  ccLang = "",
  language = "en-US",
  bufferedRange = [0, 12],
  levelEdge = 12,
  viewerIds = true,
  embeddedCaptions = false,
} = {}) {
  isVod = isVod || catchup;
  const elements = new Map();
  const getElementById = (id) => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  for (const id of ["cast-ip", "cast-overlay", "cast-state"]) getElementById(id);
  const video = getElementById("video");
  const seekable = { length: 1, start: () => 0, end: () => 12 };
  Object.assign(video, {
    currentTime: 0,
    duration: isVod ? 3600 : Infinity,
    paused: true,
    ended: false,
    readyState: 4,
    muted: false,
    volume: 1,
    videoWidth: 1280,
    videoHeight: 720,
    buffered: {
      length: bufferedRange[1] > bufferedRange[0] ? 1 : 0,
      start: () => bufferedRange[0],
      end: () => bufferedRange[1],
    },
    seekable,
    textTracks: Object.assign([], element()),
    play: async () => {
      video.paused = false;
    },
    pause: () => {
      video.paused = true;
    },
    load() {},
  });
  const pickerCalls = [];
  if (airplay) video.webkitShowPlaybackTargetPicker = () => pickerCalls.push(video.src);
  const timers = new Map();
  let timerId = 0;
  const setTimer = (fn, ms) => {
    timers.set(++timerId, { fn, ms });
    return timerId;
  };
  const calls = [];
  const beacons = [];
  const errors = [];
  let sessionNumber = 0;
  let windowStart = 0;
  let castStatus = initialCast || { active: false };
  const engines = [];
  class Hls {
    static Events = {
      MANIFEST_PARSED: "manifest",
      ERROR: "error",
      FRAG_LOADED: "fragment",
      LEVEL_SWITCHED: "level",
      SUBTITLE_TRACKS_UPDATED: "subtitles",
      LEVEL_LOADED: "levelLoaded",
      FRAG_BUFFERED: "fragBuffered",
    };
    static ErrorTypes = { NETWORK_ERROR: "network", MEDIA_ERROR: "media" };
    static isSupported() {
      return !native;
    }
    constructor(config) {
      this.config = config;
      this.handlers = new Map();
      this.subtitleTracks = [];
      this.levels = [];
      this.currentLevel = -1;
      this.loadLevel = -1;
      this.sources = [];
      engines.push(this);
    }
    on(event, handler) {
      const handlers = this.handlers.get(event) || [];
      handlers.push(handler);
      this.handlers.set(event, handlers);
    }
    off(event, handler) {
      this.handlers.set(
        event,
        (this.handlers.get(event) || []).filter((fn) => fn !== handler),
      );
    }
    emit(event, data = {}) {
      for (const handler of this.handlers.get(event) || []) handler(event, data);
    }
    loadSource(url) {
      this.sources.push(url);
      const [path, query] = url.split("?");
      const master = path.match(/\/(master|low-captions)\.m3u8$/);
      if (master) {
        // The server's master lists its single rendition, blocking when asked to.
        this.levels = ["low"].map((name) => ({
          uri:
            "http://netv.test" +
            path.replace(master[0], `/${name}.m3u8`) +
            (query === "blocking=1" ? "?blocking=1" : ""),
          details: {
            get fragments() {
              return [{ start: windowStart }];
            },
            get fragmentStart() {
              return windowStart;
            },
            edge: levelEdge,
          },
        }));
        this.currentLevel = 0;
        this.loadLevel = 0;
      }
    }
    attachMedia() {
      if (!noManifest) this.emit("manifest");
    }
    startLoad() {
      this.loads = (this.loads || 0) + 1;
    }
    recoverMediaError() {
      this.mediaRecoveries = (this.mediaRecoveries || 0) + 1;
    }
    destroy() {
      this.destroyed = true;
      this.handlers.clear();
    }
  }
  const window = Object.assign(element(), {
    // app.js, loaded before player.js on every page, defines this.
    netvNav: { region() {} },
    PLAYER_CONFIG: {
      streamRef: "ref-1",
      directUrl: "/direct/ref-1/live/stream.m3u8",
      streamType: isVod ? "movie" : "live",
      isVod,
      transcodeMode: auto ? "auto" : direct ? "never" : "always",
      liveDvrMins: isVod ? 0 : 60,
      streamId: "1",
      liveUrl: "/play/k3f9x2abqe",
      programStart,
      programEnd,
      catchup,
      catchupDays,
      catchupStart,
      catchupSeek,
      ccStyle: {},
      captionsEnabled,
      ccLang,
      sourceId: "provider",
      isHttps: false,
    },
    location: {
      href: "http://netv.test/play/live/1",
      origin: "http://netv.test",
      hostname: "netv.test",
    },
    ...(airplay ? { WebKitPlaybackTargetAvailabilityEvent: class {} } : {}),
  });
  const browserLoads = [];
  if (browserCast) {
    window.isSecureContext = true;
    window.location.protocol = "https:";
    const session = {
      loadMedia: async (request) => {
        browserLoads.push(request);
      },
      getCastDevice: () => ({ friendlyName: "Rivian" }),
      getMediaSession: () => null,
      getVolume: () => 0.5,
    };
    const castContext = {
      setOptions() {},
      addEventListener() {},
      getCurrentSession: () => session,
      requestSession: async () => {},
    };
    window.cast = {
      framework: {
        CastContext: { getInstance: () => castContext },
        CastContextEventType: { SESSION_STATE_CHANGED: "session" },
        SessionState: {},
      },
    };
    window.chrome = {
      cast: {
        AutoJoinPolicy: { ORIGIN_SCOPED: "origin" },
        media: {
          DEFAULT_MEDIA_RECEIVER_APP_ID: "default",
          MediaInfo: function (url, type) {
            this.contentId = url;
            this.contentType = type;
          },
          LoadRequest: function (media) {
            this.media = media;
          },
          GenericMediaMetadata: class {},
          StreamType: { LIVE: "LIVE", BUFFERED: "BUFFERED" },
          HlsSegmentFormat: { TS: "TS" },
          HlsVideoSegmentFormat: { MPEG2_TS: "MPEG2_TS" },
        },
      },
    };
  }
  const frames = [];
  const document = Object.assign(element(), {
    getElementById,
    createElement: element,
    head: element(),
    querySelectorAll: () => [],
    visibilityState: "visible",
  });
  const context = createContext({
    window,
    document,
    Hls,
    VTTCue: class {
      constructor(startTime, endTime, text) {
        Object.assign(this, { startTime, endTime, text });
      }
    },
    URL,
    Blob,
    AbortController,
    AbortSignal,
    Date,
    navigator: {
      language,
      sendBeacon: (url) => {
        beacons.push(url);
        return true;
      },
    },
    performance: { now: () => 1000 },
    // Frames run when a test calls them, as the browser would draw.
    requestAnimationFrame: (fn) => frames.push(fn),
    localStorage: { getItem: () => null, setItem() {} },
    setTimeout: setTimer,
    clearTimeout: (id) => timers.delete(id),
    setInterval: setTimer,
    clearInterval: (id) => timers.delete(id),
    console: {
      log() {},
      warn: (...args) => errors.push(args),
      error: (...args) => errors.push(args),
    },
    fetch: async (url, options = {}) => {
      calls.push({ url, options });
      let data = { active: false };
      if (url.startsWith("/transcode/start?")) {
        if (startReady) await startReady;
        const id = `session${++sessionNumber}`;
        data = {
          session_id: id,
          duration: isVod ? 3600 : 0,
          subtitles: [],
          seek_offset: seekOffset,
          playlist: `/transcode/${id}/${isVod ? "stream" : "low"}.m3u8${
            !isVod && startGeneration ? `?generation=${startGeneration}` : ""
          }`,
          ...(!isVod
            ? {
                ...(viewerIds ? { viewer_id: `viewer${sessionNumber}` } : {}),
                ...(embeddedCaptions
                  ? { caption_playlist: `/transcode/${id}/low-captions.m3u8` }
                  : {}),
                master_playlist: `/transcode/${id}/master.m3u8`,
                playback_buffer_seconds: playbackBufferSeconds,
              }
            : {}),
        };
      } else if (options.method === "DELETE") {
        if (stopReady) await stopReady;
        if (stopFailure) return { ok: false, status: 503 };
      } else if (url.startsWith("/transcode/progress/")) {
        data = { duration: transcodedDuration };
      } else if (url.includes("/health")) {
        data = {
          playlist: `/transcode/session${sessionNumber}/low.m3u8${
            startGeneration ? `?generation=${startGeneration}` : ""
          }`,
        };
      } else if (url.startsWith("/api/live/program/")) {
        data = nextProgram || { title: "", desc: "", start: 0, end: 0 };
      } else if (url === "/api/cast/status") {
        if (statusReady) await statusReady;
        data = castStatus;
      } else if (url === "/api/cast/devices") {
        data = { devices: [] };
      } else if (url === "/api/cast/start") {
        if (castFailure)
          return {
            ok: false,
            json: async () => ({ detail: "TV could not load the stream" }),
          };
        castStatus = {
          active: true,
          name: "Living room",
          host: "192.168.1.50",
          state: "PLAYING",
          volume: 0.5,
          session_id: JSON.parse(options.body).session_id,
        };
        data = castStatus;
      } else if (url === "/api/cast/control") {
        if (JSON.parse(options.body).action === "stop") castStatus = { active: false };
        data = castStatus;
      }
      return { ok: true, json: async () => data };
    },
  });
  for (const script of ["live-playback.js", "cast.js", "player.js"]) {
    runInContext(
      readFileSync(join(__dirname, "../../netv/static/js", script), "utf8"),
      context,
    );
  }
  await flush();
  return {
    video,
    window,
    document,
    elements,
    frames,
    engines,
    calls,
    beacons,
    errors,
    timers,
    pickerCalls,
    browserLoads,
    setWindowStart: (value) => {
      windowStart = value;
    },
    setBuffered: (start, end) => {
      bufferedRange = [start, end];
      video.buffered.length = end > start ? 1 : 0;
    },
    setTranscodedDuration: (value) => {
      transcodedDuration = value;
    },
    setLiveEdge: (value) => {
      for (const engine of engines) engine.liveSyncPosition = value;
    },
    async tickProgram() {
      const [, timer] = [...timers].find(([, timer]) => timer.ms === 1000);
      await timer.fn();
      await flush();
    },
    // Runs the newest timer of this delay, as the player just scheduled it.
    async fireTimer(ms) {
      const [id, timer] = [...timers].filter(([, timer]) => timer.ms === ms).at(-1);
      timers.delete(id);
      await timer.fn();
      await flush();
    },
    async pollHealth() {
      const [id, timer] = [...timers].find(([, timer]) => timer.ms === 2000);
      timers.delete(id);
      await timer.fn();
    },
  };
}

test("web live startup and health polls use one shared adaptive session", async () => {
  const p = await player();
  const start = p.calls.find((call) => call.url.startsWith("/transcode/start?"));
  const params = new URL(start.url, "http://netv.test").searchParams;
  assert.equal(params.get("content_type"), "live");
  // The provider's URL never reaches the page: the server knows it by this ref.
  assert.equal(params.get("ref"), "ref-1");
  assert.equal(params.get("url"), null);
  // The player waits for its own lead, so the server answers at the first segment.
  assert.equal(params.get("min_segments"), "1");
  assert.equal(p.engines.length, 1);
  const hls = p.engines[0];
  // The web player reloads its playlist by blocking, so new segments arrive at once.
  assert.deepEqual(hls.sources, ["/transcode/session1/master.m3u8?blocking=1"]);
  assert.equal(hls.config.lowLatencyMode, true);
  assert.equal(hls.config.startPosition, -1);
  assert.equal(hls.config.liveSyncDuration, 12);
  assert.equal(hls.config.maxBufferLength, 30);
  await p.pollHealth();
  // Health polls name this page, keeping it a viewer of the shared session.
  assert.ok(
    p.calls.some((call) => call.url === "/transcode/session1/health?viewer=viewer1"),
  );
  // Health feedback for the same generation leaves playback alone.
  assert.equal(hls.nextLevel, undefined);
  assert.equal(p.engines.length, 1);
  assert.equal(
    p.calls.filter((call) => call.url.startsWith("/transcode/start?")).length,
    1,
  );
  assert.equal(
    p.calls.some((call) => call.url.includes("/progress/")),
    false,
  );
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, ["/transcode/session1/stop?force=true&viewer=viewer1"]);
  assert.deepEqual(p.errors, []);
});

// A broadcast's CEA-608 channels as hls.js names them: CC1 English, CC3 unlabeled.
const broadcastCaptions = () => [
  { kind: "captions", label: "Unknown CC", language: "", mode: "showing" },
  { kind: "captions", label: "English", language: "en", mode: "showing" },
];
const captionModes = (video) => Array.from(video.textTracks, (track) => track.mode);

test("captions follow the browser language when none is chosen", async () => {
  const p = await player({ captionsEnabled: true, language: "en-US" });
  p.video.textTracks.push(...broadcastCaptions());
  // Chrome switched the second channel on by itself.
  await p.video.textTracks.emit("change");
  assert.deepEqual(captionModes(p.video), ["hidden", "showing"]);
  assert.deepEqual(p.errors, []);
});

test("a chosen caption language wins over the browser language", async () => {
  const p = await player({
    captionsEnabled: true,
    ccLang: "en",
    language: "es-MX",
  });
  p.video.textTracks.push(...broadcastCaptions());
  await p.video.textTracks.emit("change");
  assert.deepEqual(captionModes(p.video), ["hidden", "showing"]);
});

test("one showing caption track is the viewer's pick and stays", async () => {
  const p = await player({ captionsEnabled: true, language: "en-US" });
  const tracks = broadcastCaptions();
  tracks[1].mode = "hidden";
  p.video.textTracks.push(...tracks);
  await p.video.textTracks.emit("change");
  assert.deepEqual(captionModes(p.video), ["showing", "hidden"]);
});

test("a track turned on from the browser menu with CC off stays on", async () => {
  // The native or fullscreen caption menu sets the mode directly.
  const p = await player({ captionsEnabled: false });
  const tracks = broadcastCaptions();
  tracks[0].mode = "hidden";
  p.video.textTracks.push(...tracks);
  await p.video.textTracks.emit("change");
  assert.deepEqual(captionModes(p.video), ["hidden", "showing"]);
  assert.equal(p.elements.get("toggle-cc").classList.contains("active"), true);
});

test("with CC off, two showing tracks are all hidden", async () => {
  const p = await player({ captionsEnabled: false });
  p.video.textTracks.push(...broadcastCaptions());
  await p.video.textTracks.emit("change");
  assert.deepEqual(captionModes(p.video), ["hidden", "hidden"]);
});

test("against a server without viewer ids, health and stop name no viewer", async () => {
  // The new player against an older server: no viewer_id in the start response.
  const p = await player({ viewerIds: false });
  await p.pollHealth();
  assert.ok(p.calls.some((call) => call.url === "/transcode/session1/health"));
  assert.equal(
    p.calls.some((call) => call.url.includes("viewer")),
    false,
  );
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, ["/transcode/session1/stop?force=true"]);
  assert.deepEqual(p.errors, []);
});

test("a stream with embedded captions keeps CC available before the first caption", async () => {
  // hls.js creates the caption track only when it decodes the first cue.
  const p = await player({ embeddedCaptions: true });
  await p.fireTimer(1000);
  p.engines[0].emit("subtitles");
  assert.equal(p.elements.get("toggle-cc").disabled, false);
  assert.deepEqual(p.errors, []);
});

test("a stream with embedded captions plays the master that declares them", async () => {
  const p = await player({ embeddedCaptions: true });
  assert.deepEqual(p.engines[0].sources, [
    "/transcode/session1/low-captions.m3u8?blocking=1",
  ]);
  assert.deepEqual(p.errors, []);
});

test("native playback of embedded captions keeps CC and turns it on", async () => {
  const p = await player({
    native: true,
    embeddedCaptions: true,
    captionsEnabled: true,
  });
  assert.equal(p.video.src, "/transcode/session1/low-captions.m3u8");
  await p.video.emit("loadedmetadata");
  assert.equal(p.elements.get("toggle-cc").disabled, false);
  const track = Object.assign(element(), {
    kind: "captions",
    label: "CC1",
    language: "en",
    mode: "hidden",
  });
  p.video.textTracks.push(track);
  await p.video.textTracks.emit("addtrack", { track });
  assert.equal(track.mode, "showing");
  assert.deepEqual(p.errors, []);
});

test("a stream without captions disables CC once none appeared", async () => {
  const p = await player();
  await p.fireTimer(1000);
  assert.equal(p.elements.get("toggle-cc").disabled, true);
});

test("live playback waits until the media buffer leads by a segment", async () => {
  // Real-time output: playing from the first fragment stalls on the next full segment.
  const p = await player({ bufferedRange: [0.6, 0.7], levelEdge: 0 });
  const buffered = (end) => {
    p.setBuffered(0.6, end);
    p.engines[0].emit("fragBuffered");
  };
  assert.equal(p.video.paused, true);
  // Lead counts from where playback will begin: the start of the buffer.
  buffered(1.7);
  assert.equal(p.video.paused, true);
  // The paused playhead moves into the buffer: hls.js skips gaps only while playing,
  // and loads nothing more for a playhead stuck before its first fragment.
  assert.equal(p.video.currentTime, 0.6);
  buffered(1.85);
  assert.equal(p.video.paused, false);
  // Later fragments do not restart anything.
  p.video.pause();
  buffered(3);
  assert.equal(p.video.paused, true);
  assert.deepEqual(p.errors, []);
});

test("live playback starts anyway once the playlist leads by seconds", async () => {
  // A source whose audio and video never overlap leaves the media buffer empty.
  const p = await player({ bufferedRange: [0, 0], levelEdge: 0 });
  const listed = (edge) =>
    p.engines[0].emit("levelLoaded", { details: { fragmentStart: 0, edge } });
  listed(2.9);
  assert.equal(p.video.paused, true);
  listed(3);
  assert.equal(p.video.paused, false);
});

test("a session already holding a segment of lead plays at once", async () => {
  const p = await player({ levelEdge: 30 });
  assert.equal(p.video.paused, false);
});

test("VOD plays as soon as its manifest is parsed", async () => {
  const p = await player({ isVod: true, bufferedRange: [0, 0.1] });
  assert.equal(p.video.paused, false);
});

test("web live playback honors the measured buffer target", async () => {
  const p = await player({ playbackBufferSeconds: 40 });
  assert.equal(p.engines[0].config.liveSyncDuration, 40);
  assert.equal(p.engines[0].config.maxBufferLength, 40);
  assert.deepEqual(p.errors, []);
});

test("HTTP player casts a shared session and leaves it running after page close", async () => {
  const p = await player();
  p.elements.get("cast-ip").value = "192.168.1.50";
  await p.elements.get("cast-form").emit("submit", { preventDefault() {} });
  await flush();
  const request = p.calls.find((call) => call.url === "/api/cast/start");
  assert.equal(JSON.parse(request.options.body).session_id, "session1");
  assert.equal(JSON.parse(request.options.body).server_url, "http://netv.test");
  assert.equal(p.video.paused, true);
  assert.equal(p.elements.get("cast-overlay").classList.contains("hidden"), false);
  assert.equal(p.elements.get("cast-picker").classList.contains("hidden"), true);
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, []);
  assert.equal(
    p.calls.some((call) => call.options.method === "DELETE"),
    false,
  );
});

test("nearby browser casting uses the public origin and retains playback after page close", async () => {
  const p = await player({ browserCast: true });
  await p.elements.get("cast-browser").emit("click");
  await flush();
  assert.equal(p.browserLoads.length, 1);
  assert.equal(
    p.browserLoads[0].media.contentId,
    "http://netv.test/transcode/session1/master.m3u8",
  );
  assert.equal(p.browserLoads[0].media.streamType, "LIVE");
  assert.equal(p.video.paused, true);
  assert.equal(p.engines[0].destroyed, true);
  assert.equal(
    p.calls.some((call) => call.url === "/api/cast/start"),
    false,
  );
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, []);
  assert.equal(
    p.calls.some((call) => call.options.method === "DELETE"),
    false,
  );
});

test("nearby VOD casting sends the HLS-relative position and buffered stream type", async () => {
  const p = await player({ browserCast: true, isVod: true, seekOffset: 120 });
  p.video.currentTime = 25;
  await p.elements.get("cast-browser").emit("click");
  await flush();
  assert.equal(
    p.browserLoads[0].media.contentId,
    "http://netv.test/transcode/session1/stream.m3u8",
  );
  assert.equal(p.browserLoads[0].media.streamType, "BUFFERED");
  assert.equal(p.browserLoads[0].currentTime, 25);
});

test("nearby casting prepares HLS when starting from direct playback", async () => {
  const p = await player({ browserCast: true, direct: true });
  await p.elements.get("cast-browser").emit("click");
  await flush();
  assert.equal(
    p.browserLoads[0].media.contentId,
    "http://netv.test/transcode/session1/master.m3u8",
  );
  assert.equal(
    p.calls.filter((call) => call.url.startsWith("/transcode/start?")).length,
    1,
  );
});

test("failed casting keeps local playback and exposes the receiver error", async () => {
  const p = await player({ castFailure: true });
  p.elements.get("cast-ip").value = "192.168.1.50";
  await p.elements.get("cast-form").emit("submit", { preventDefault() {} });
  await flush();
  assert.equal(
    p.elements.get("cast-error").textContent,
    "TV could not load the stream",
  );
  assert.equal(p.elements.get("cast-overlay").classList.contains("hidden"), true);
  assert.equal(p.video.paused, false);
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, ["/transcode/session1/stop?force=true&viewer=viewer1"]);
});

test("direct live playback prepares local HLS before casting", async () => {
  const p = await player({ direct: true });
  p.elements.get("cast-ip").value = "192.168.1.50";
  await p.elements.get("cast-form").emit("submit", { preventDefault() {} });
  await flush();
  const startIndex = p.calls.findIndex((call) =>
    call.url.startsWith("/transcode/start?"),
  );
  const castIndex = p.calls.findIndex((call) => call.url === "/api/cast/start");
  assert.ok(startIndex >= 0 && castIndex > startIndex);
  assert.equal(p.video.paused, true);
});

test("device discovery offers manual fallback and controls use authenticated API", async () => {
  const p = await player();
  await p.elements.get("cast-btn").emit("click", { stopPropagation() {} });
  await flush();
  assert.match(p.elements.get("cast-discovery").textContent, /No devices found/);
  p.elements.get("cast-ip").value = "192.168.1.50";
  await p.elements.get("cast-form").emit("submit", { preventDefault() {} });
  await flush();
  await p.elements.get("cast-pause").emit("click");
  await flush();
  assert.deepEqual(JSON.parse(p.calls.at(-1).options.body), {
    action: "pause",
  });
  await p.elements.get("cast-stop").emit("click");
  await flush();
  assert.equal(p.elements.get("cast-state").textContent, "Casting has ended.");
  assert.equal(p.elements.get("cast-local").textContent, "Play here");
});

test("returning to a player restores cast controls without starting local playback", async () => {
  const p = await player({
    initialCast: {
      active: true,
      name: "TV",
      state: "PLAYING",
      volume: 0.5,
      session_id: "existing-cast",
    },
  });
  assert.equal(p.elements.get("cast-overlay").classList.contains("hidden"), false);
  assert.equal(p.video.paused, true);
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/start?")),
    false,
  );
});

test("local startup waits for delayed cast status instead of consuming another stream slot", async () => {
  let resolve;
  const statusReady = new Promise((done) => {
    resolve = done;
  });
  const p = await player({
    statusReady,
    initialCast: {
      active: true,
      name: "TV",
      state: "PLAYING",
      volume: 0.5,
      session_id: "existing-cast",
    },
  });
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/start?")),
    false,
  );
  resolve();
  await flush();
  assert.equal(p.elements.get("cast-overlay").classList.contains("hidden"), false);
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/start?")),
    false,
  );
});

test("movie handoff sends the HLS-relative position rather than adding the seek offset", async () => {
  const p = await player({ isVod: true, seekOffset: 120 });
  p.video.currentTime = 90;
  p.elements.get("cast-ip").value = "192.168.1.50";
  await p.elements.get("cast-form").emit("submit", { preventDefault() {} });
  await flush();
  const request = p.calls.find((call) => call.url === "/api/cast/start");
  assert.equal(JSON.parse(request.options.body).current_time, 90);
  assert.equal(p.video.paused, true);
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, []);
});

test("web restart stops the live session before starting a replacement", async () => {
  const p = await player();
  await p.elements.get("menu-restart").emit("click");
  await flush();
  const stop = p.calls.findIndex(
    (call) => call.url === "/transcode/session1?force=true&viewer=viewer1",
  );
  const starts = p.calls.flatMap((call, index) =>
    call.url.startsWith("/transcode/start?") ? [index] : [],
  );
  assert.equal(starts.length, 2);
  assert.ok(stop > starts[0] && stop < starts[1]);
  assert.equal(p.engines[0].destroyed, true);
  assert.deepEqual(p.engines[1].sources, [
    "/transcode/session2/master.m3u8?blocking=1",
  ]);
  assert.deepEqual(p.errors, []);
});

test("web VOD keeps its existing media playlist, seek configuration, and progress polls", async () => {
  const p = await player({ isVod: true });
  assert.equal(p.calls[0].url.includes("fast_start"), false);
  assert.equal(p.calls[0].url.includes("min_segments"), false);
  assert.deepEqual(p.engines[0].sources, ["/transcode/session1/stream.m3u8"]);
  assert.equal(p.engines[0].config.startPosition, 0);
  assert.equal(
    p.calls.some((call) => call.url.includes("/progress/")),
    true,
  );
  assert.equal(
    p.calls.some((call) => call.url.includes("/health")),
    false,
  );
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, ["/transcode/session1/stop?force=false"]);
  assert.deepEqual(p.errors, []);
});

test("native HLS stays on low while maintaining the shared live session", async () => {
  const p = await player({ native: true });
  assert.equal(p.engines.length, 0);
  assert.equal(p.video.src, "/transcode/session1/low.m3u8");
  assert.equal(
    p.calls.some((call) => call.url.includes("/health")),
    true,
  );
  assert.deepEqual(p.errors, []);
});

test("native HLS recognizes a recovered fast-start session", async () => {
  const p = await player({ native: true, startGeneration: 3 });
  assert.equal(p.video.src, "/transcode/session1/low.m3u8?generation=3");
  assert.equal(
    p.calls.some((call) => call.url.includes("/health")),
    true,
  );
  assert.deepEqual(p.errors, []);
});

test("direct playback does not opt into server transcoding or adaptive health", async () => {
  const p = await player({ direct: true });
  // Played through the server's relay, never from the provider itself.
  assert.deepEqual(p.engines[0].sources, ["/direct/ref-1/live/stream.m3u8"]);
  assert.equal(p.engines[0].config.liveSyncDurationCount, 3);
  assert.deepEqual(
    p.calls.map((call) => call.url),
    ["/api/cast/status"],
  );
  assert.deepEqual(p.errors, []);
});

test("returning from the browser back-forward cache can restart playback", async () => {
  const p = await player();
  await p.window.emit("pagehide");
  await p.window.emit("pageshow", { persisted: true });
  await flush();
  assert.equal(p.engines.length, 2);
  assert.deepEqual(p.engines[1].sources, [
    "/transcode/session2/master.m3u8?blocking=1",
  ]);
  assert.equal(p.engines[0].destroyed, true);
  assert.deepEqual(p.errors, []);
});

test("live DVR resumes at the oldest available position when the pause expired", async () => {
  const p = await player();
  await p.video.emit("pause");
  p.video.currentTime = 120;
  p.setWindowStart(300);
  await p.video.emit("play");
  assert.equal(p.video.currentTime, 300.1);
  assert.deepEqual(p.errors, []);
});

test("live DVR keeps the paused position when still within the window", async () => {
  const p = await player();
  await p.video.emit("pause");
  p.video.currentTime = 350;
  p.setWindowStart(300);
  await p.video.emit("play");
  assert.equal(p.video.currentTime, 350);
  assert.deepEqual(p.errors, []);
});

test("live playback before any pause does not seek to the window start", async () => {
  const p = await player();
  p.video.currentTime = 0;
  p.setWindowStart(300);
  await p.video.emit("play");
  assert.equal(p.video.currentTime, 0);
  assert.deepEqual(p.errors, []);
});

test("native HLS DVR resumes at the oldest available position when the pause expired", async () => {
  const p = await player({ native: true });
  await p.video.emit("pause");
  p.video.currentTime = 120;
  p.video.seekable.start = () => 300;
  await p.video.emit("play");
  assert.equal(p.video.currentTime, 300.1);
  assert.deepEqual(p.errors, []);
});

test("live controls span the EPG program instead of the segment window", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 600.5,
    programEnd: now + 1199.5,
  });
  p.setLiveEdge(0);
  await p.video.emit("timeupdate");
  assert.equal(
    p.elements.get("progress-container").classList.contains("hidden"),
    false,
  );
  assert.equal(p.elements.get("time-current").textContent, "10:00");
  assert.equal(p.elements.get("time-duration").textContent, "-19:59");
  assertPercent(p.elements.get("progress-played").style.width, (600.5 / 1800) * 100);
  assert.deepEqual(p.errors, []);
});

test("rewinding moves the program position back by the distance behind live", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 600.5,
    programEnd: now + 1199.5,
  });
  p.setLiveEdge(300);
  p.video.currentTime = 180;
  await p.video.emit("timeupdate");
  assert.equal(p.elements.get("time-current").textContent, "8:00");
  assert.equal(p.elements.get("time-duration").textContent, "-21:59");
  assert.deepEqual(p.errors, []);
});

test("live DVR window is shaded over the part of the program still held", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 600.5,
    programEnd: now + 1199.5,
  });
  p.setLiveEdge(300);
  p.setWindowStart(60);
  await p.video.emit("timeupdate");
  const buffered = p.elements.get("progress-buffered");
  assertPercent(buffered.style.left, ((600.5 - 240) / 1800) * 100);
  assertPercent(buffered.style.width, (240 / 1800) * 100);
  assert.deepEqual(p.errors, []);
});

test("clicking the live bar seeks to that moment of the broadcast", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 600.5,
    programEnd: now + 1199.5,
  });
  p.setLiveEdge(1000);
  await p.elements.get("progress-bar").emit("click", { clientX: 25 });
  // 25% into a 30 minute program is 450s, which is 150.5s before now.
  assert.ok(
    Math.abs(p.video.currentTime - 849.5) < 1,
    `seeked to ${p.video.currentTime}`,
  );
  assert.deepEqual(p.errors, []);
});

test("clicking before the DVR window clamps to the oldest retained position", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 600.5,
    programEnd: now + 1199.5,
  });
  p.setLiveEdge(1000);
  p.setWindowStart(900);
  await p.elements.get("progress-bar").emit("click", { clientX: 1 });
  assert.equal(p.video.currentTime, 900.1);
  assert.deepEqual(p.errors, []);
});

test("live controls stay hidden when the guide has no program", async () => {
  const p = await player();
  assert.equal(p.elements.get("progress-container").classList.contains("hidden"), true);
  assert.deepEqual(p.errors, []);
});

test("archive scrubbing previews the full program and reopens only on release", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
  });
  const slider = p.elements.get("progress-seek");
  const originalUrl = p.window.location.href;
  slider.value = "25";
  await slider.emit("input");
  slider.value = "75";
  await slider.emit("input");
  await p.video.emit("timeupdate");
  assert.equal(p.window.location.href, originalUrl);
  assert.equal(p.video.currentTime, 0);
  assert.equal(p.elements.get("time-current").textContent, "45:00");
  assert.equal(slider.getAttribute("aria-valuetext"), "45:00");
  assertPercent(p.elements.get("progress-played").style.width, 75);
  await slider.emit("change");
  assert.equal(p.window.location.href, `/play/k3f9x2abqe?start=${start + 2700}`);
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/seek/")),
    false,
  );
  assert.deepEqual(p.errors, []);
});

test("archive scrubbing backward before the opened archive requests that earlier moment", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    catchupStart: start + 1800,
    transcodedDuration: 300,
  });
  const slider = p.elements.get("progress-seek");
  slider.value = "10";
  await slider.emit("input");
  await slider.emit("change");
  assert.equal(p.window.location.href, `/play/k3f9x2abqe?start=${start + 360}`);
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/seek/")),
    false,
  );
});

test("archive scrubbing within transcoded video seeks locally in either direction", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    catchupStart: start + 600,
    transcodedDuration: 1200,
  });
  const originalUrl = p.window.location.href;
  const slider = p.elements.get("progress-seek");
  for (const [percent, position] of [
    [40, 840],
    [20, 120],
  ]) {
    slider.value = String(percent);
    await slider.emit("input");
    await slider.emit("change");
    assert.equal(p.video.currentTime, position);
    assert.equal(p.window.location.href, originalUrl);
  }
  assert.deepEqual(p.errors, []);
});

test("archive jump times remain relative to the full program after seeking", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  for (const [text, seconds] of [
    ["10:00", 600],
    ["1:30:00", 5400],
  ]) {
    const p = await player({
      catchup: true,
      programStart: start,
      programEnd: start + 7200,
      catchupStart: start + 1800,
      transcodedDuration: 120,
    });
    const input = p.elements.get("seek-input");
    input.value = text;
    await input.emit("keydown", { key: "Enter", preventDefault() {} });
    assert.equal(p.window.location.href, `/play/k3f9x2abqe?start=${start + seconds}`);
    assert.equal(input.classList.contains("ring-red-500"), false);
  }
});

test("archive seek waits for a forced stop before navigating", async () => {
  let release;
  const stopReady = new Promise((resolve) => {
    release = resolve;
  });
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    stopReady,
  });
  const originalUrl = p.window.location.href;
  const seek = p.elements.get("progress-bar").emit("click", { clientX: 50 });
  await flush();
  assert.equal(p.window.location.href, originalUrl);
  assert.ok(
    p.calls.some(
      (call) =>
        call.options.method === "DELETE" &&
        call.url === "/transcode/session1?force=true",
    ),
  );
  release();
  await seek;
  assert.equal(p.window.location.href, `/play/k3f9x2abqe?start=${start + 1800}`);
  assert.deepEqual(p.errors, []);
});

test("archive seek during startup releases the eventual session before navigating", async () => {
  let finishStartup;
  const startReady = new Promise((resolve) => {
    finishStartup = resolve;
  });
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    startReady,
  });
  const originalUrl = p.window.location.href;
  const seek = p.elements.get("progress-bar").emit("click", { clientX: 50 });
  await flush();
  assert.equal(p.window.location.href, originalUrl);
  finishStartup();
  await seek;
  assert.equal(p.window.location.href, `/play/k3f9x2abqe?start=${start + 1800}`);
  assert.ok(p.calls.some((call) => call.url === "/transcode/session1?force=true"));
});

test("failed archive stop blocks navigation and exposes a playback error", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    stopFailure: true,
  });
  const originalUrl = p.window.location.href;
  await p.elements.get("progress-bar").emit("click", { clientX: 50 });
  assert.equal(p.window.location.href, originalUrl);
  assert.equal(p.elements.get("error").classList.contains("hidden"), false);
  assert.match(String(p.errors[0]), /Could not release the previous stream/);
});

test("closing archive playback force-releases its upstream slot", async () => {
  const p = await player({ catchup: true });
  await p.window.emit("pagehide");
  assert.deepEqual(p.beacons, ["/transcode/session1/stop?force=true"]);
});

test("go live releases archive playback before navigating", async () => {
  const p = await player({ catchup: true });
  await p.elements.get("go-live-btn").emit("click", { preventDefault() {} });
  await flush();
  assert.ok(p.calls.some((call) => call.url === "/transcode/session1?force=true"));
  assert.equal(p.window.location.href, "/play/k3f9x2abqe");
});

test("user archive seek cancels a pending sub-minute startup skip", async () => {
  const start = Math.floor(Date.now() / 1000) - 7200;
  const p = await player({
    catchup: true,
    programStart: start,
    programEnd: start + 3600,
    catchupSeek: 25,
    transcodedDuration: 10,
  });
  await p.elements.get("progress-bar").emit("click", { clientX: 0 });
  p.setTranscodedDuration(120);
  await p.pollHealth();
  assert.equal(p.video.currentTime, 0);
  assert.deepEqual(p.errors, []);
});

test("cancelling a scrub restores the displayed position without seeking", async () => {
  const p = await player({ isVod: true });
  p.video.currentTime = 120;
  const slider = p.elements.get("progress-seek");
  slider.value = "75";
  await slider.emit("input");
  await slider.emit("pointercancel");
  assert.equal(p.video.currentTime, 120);
  assert.equal(p.elements.get("time-current").textContent, "2:00");
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/seek/")),
    false,
  );
});

test("scrubbing regular VOD preserves local and server-side seeking", async () => {
  const p = await player({ isVod: true, transcodedDuration: 1200 });
  const slider = p.elements.get("progress-seek");
  slider.value = "25";
  await slider.emit("input");
  await slider.emit("change");
  assert.equal(p.video.currentTime, 900);
  assert.equal(
    p.calls.some((call) => call.url.startsWith("/transcode/seek/")),
    false,
  );
  slider.value = "75";
  await slider.emit("input");
  await slider.emit("change");
  assert.ok(p.calls.some((call) => call.url === "/transcode/seek/session1?time=2700"));
  assert.deepEqual(p.errors, []);
});

test("the next guide program takes over when the current one ends", async () => {
  const now = Date.now() / 1000;
  const p = await player({
    programStart: now - 1800,
    programEnd: now - 0.5,
    nextProgram: {
      title: "Next Up",
      desc: "Later tonight",
      start: now - 0.5,
      end: now + 3599.5,
    },
  });
  p.setLiveEdge(0);
  await p.tickProgram();
  assert.equal(p.elements.get("program-title").textContent, " \u2014 Next Up");
  assert.equal(p.elements.get("program-desc").textContent, "Later tonight");
  assert.equal(p.elements.get("program-desc").classList.contains("hidden"), false);
  assert.equal(p.elements.get("time-current").textContent, "0:00");
  assert.equal(p.elements.get("time-duration").textContent, "-59:59");
  assert.deepEqual(p.errors, []);
});

const transcodeStarts = (p) =>
  p.calls.filter((call) => call.url.startsWith("/transcode/start?")).length;

test("AirPlay stays hidden outside Safari", async () => {
  const p = await player();
  assert.equal(p.elements.get("airplay-btn").classList.contains("hidden"), true);
});

test("AirPlay moves an hls.js session to native HLS within the click gesture", async () => {
  const p = await player({ airplay: true });
  assert.equal(p.elements.get("airplay-btn").classList.contains("hidden"), false);
  p.video.disableRemotePlayback = true;
  await p.elements.get("airplay-btn").emit("click", { stopPropagation() {} });
  assert.equal(p.engines[0].destroyed, true);
  assert.equal(p.video.disableRemotePlayback, false);
  assert.deepEqual(p.pickerCalls, ["/transcode/session1/low.m3u8"]);
  assert.equal(transcodeStarts(p), 1);
  await p.elements.get("airplay-btn").emit("click", { stopPropagation() {} });
  assert.equal(p.pickerCalls.length, 2);
  assert.equal(p.engines.length, 1);
  assert.deepEqual(p.errors, []);
});

test("AirPlay keeps the VOD position when switching players", async () => {
  const p = await player({ airplay: true, isVod: true });
  p.video.currentTime = 120;
  await p.elements.get("airplay-btn").emit("click", { stopPropagation() {} });
  p.video.currentTime = 0;
  await p.video.emit("loadedmetadata");
  assert.equal(p.video.currentTime, 120);
  assert.deepEqual(p.pickerCalls, ["/transcode/session1/stream.m3u8"]);
});

test("AirPlay prepares a neTV stream for direct playback, then opens the picker", async () => {
  const p = await player({ airplay: true, direct: true });
  assert.equal(transcodeStarts(p), 0);
  await p.elements.get("airplay-btn").emit("click", { stopPropagation() {} });
  await flush();
  assert.equal(transcodeStarts(p), 1);
  assert.equal(p.engines.length, 1);
  assert.equal(p.engines[0].destroyed, true);
  assert.equal(p.video.src, "/transcode/session1/low.m3u8");
  assert.deepEqual(p.pickerCalls, []);
  assert.match(p.elements.get("airplay-status").textContent, /Tap AirPlay again/);
  await p.elements.get("airplay-btn").emit("click", { stopPropagation() {} });
  assert.deepEqual(p.pickerCalls, ["/transcode/session1/low.m3u8"]);
});

const isHidden = (p, id) => p.elements.get(id).classList.contains("hidden");
const lose = (engine) =>
  engine.emit("error", { fatal: true, type: "network", response: { code: 0 } });

test("live playback reconnects with backoff through a provider outage", async () => {
  const p = await player();
  lose(p.engines[0]);
  assert.equal(p.engines[0].destroyed, true);
  assert.equal(isHidden(p, "error"), true);
  assert.equal(isHidden(p, "loading"), false);
  await p.fireTimer(1000);
  assert.equal(transcodeStarts(p), 2);
  lose(p.engines[1]);
  await p.fireTimer(2000);
  assert.equal(transcodeStarts(p), 3);
  p.engines[2].emit("fragment");
  assert.equal(isHidden(p, "loading"), true);
  lose(p.engines[2]);
  await p.fireTimer(1000);
  assert.equal(transcodeStarts(p), 4, "backoff resets after recovery");
  assert.equal(isHidden(p, "error"), true);
});

test("live playback gives up after two minutes without the stream", async () => {
  const p = await player();
  const now = Date.now;
  let clock = now();
  Date.now = () => clock;
  try {
    lose(p.engines[0]);
    await p.fireTimer(1000);
    clock += 121000;
    lose(p.engines[1]);
    assert.equal(isHidden(p, "error"), false);
  } finally {
    Date.now = now;
  }
});

test("direct live playback reconnects through the relay", async () => {
  const p = await player({ direct: true });
  lose(p.engines[0]);
  await p.fireTimer(1000);
  assert.equal(transcodeStarts(p), 0);
  assert.deepEqual(p.engines[1].sources, ["/direct/ref-1/live/stream.m3u8"]);
  assert.equal(isHidden(p, "error"), true);
});

test("auto mode falls back to transcoding when direct playback fails mid-stream", async () => {
  const p = await player({ auto: true });
  assert.equal(transcodeStarts(p), 0);
  lose(p.engines[0]);
  await flush();
  assert.equal(transcodeStarts(p), 1);
  assert.equal(isHidden(p, "error"), true);
});

test("a refused live source shows the error without retrying", async () => {
  const p = await player({ direct: true, noManifest: true });
  const timers = p.timers.size;
  p.engines[0].emit("error", {
    fatal: true,
    type: "network",
    response: { code: 403 },
  });
  assert.equal(p.engines[0].destroyed, true);
  assert.equal(isHidden(p, "error"), false);
  assert.equal(p.timers.size, timers);
});

test("VOD network errors retry in place with backoff", async () => {
  const p = await player({ isVod: true });
  lose(p.engines[0]);
  assert.equal(p.engines[0].destroyed, undefined);
  await p.fireTimer(2000);
  assert.equal(p.engines[0].loads, 1);
  assert.equal(isHidden(p, "error"), true);
});

// A CEA-608 screen as hls.js hands it to the cue handler: 15 rows of 32 cells.
function captionScreen(...lines) {
  const rows = Array.from({ length: 15 }, () => ({ chars: [] }));
  lines.forEach((line, i) => {
    rows[13 - lines.length + 1 + i].chars = [...line].map((uchar) => ({ uchar }));
  });
  return { rows };
}

function captionTrack() {
  const cues = [];
  return {
    cues,
    addCue: (cue) => cues.push(cue),
    removeCue: (cue) => cues.splice(cues.indexOf(cue), 1),
    shownAt: (t) =>
      cues
        .filter((cue) => cue.startTime <= t && t < cue.endTime)
        .map((cue) => cue.text),
  };
}

test("roll-up captions show each line at its broadcast time without gaps", async () => {
  const p = await player();
  const { cueHandler } = p.engines[0].config;
  const track = captionTrack();
  // hls.js emits one cue per line, each screen the previous line plus the new one.
  const screens = [["A"], ["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]];
  for (const [i, lines] of screens.entries()) {
    cueHandler.newCue(track, i, i + 1, captionScreen(...lines));
  }
  const shown = (t) => track.shownAt(t);
  assert.deepEqual(shown(0.5), ["A"]);
  assert.deepEqual(shown(1.5), ["A\nB"]);
  // A new line shows the moment it is broadcast, not when its page fills.
  assert.deepEqual(shown(2.5), ["B\nC"]);
  assert.deepEqual(shown(3.5), ["C\nD"]);
  assert.deepEqual(shown(4.5), ["D\nE"]);
  // No caption follows E: D E holds 6 s past its end.
  assert.deepEqual(shown(10.5), ["D\nE"]);
  assert.deepEqual(shown(11.5), []);
  for (let t = 0; t < 11; t += 0.1) assert.equal(shown(t).length, 1, `t=${t}`);
});

test("captions roll up one row at a time instead of redrawing the block", async () => {
  const p = await player({ captionsEnabled: true });
  const drawn = [];
  const box = p.elements.get("caption-box");
  box.replaceChildren = (...lines) => drawn.push(lines.map((line) => line.textContent));
  const overlay = p.elements.get("caption-overlay");
  const track = Object.assign(element(), {
    kind: "captions",
    label: "English",
    language: "en",
    mode: "showing",
    activeCues: [],
  });
  p.video.textTracks.push(track);
  await p.video.textTracks.emit("addtrack", { track });
  const show = async (...lines) => {
    track.activeCues = lines.length ? [{ text: lines.join("\n") }] : [];
    await track.emit("cuechange");
  };
  await show("A");
  await show("A", "<i>B</i>");
  assert.deepEqual(drawn.at(-1), ["A", "B"]);
  assert.equal(overlay.style.height, "2.7em");
  // The new line rolls in: the old screen is drawn above it and slides up one row,
  // so the line being read moves up smoothly rather than leaping to the top.
  await show("B", "C");
  assert.deepEqual(drawn.at(-1), ["A", "B", "C"]);
  assert.equal(box.style.transform, "translateY(0)");
  for (const frame of p.frames.splice(0)) frame();
  assert.equal(box.style.transform, "translateY(-1.35em)");
  assert.equal(overlay.style.height, "2.7em");
  // A new screen that is no roll (a pop-on caption) is drawn as it is.
  await show("X");
  assert.deepEqual(drawn.at(-1), ["X"]);
  assert.equal(overlay.classList.contains("hidden"), false);
  await show();
  assert.equal(overlay.classList.contains("hidden"), true);
  // The browser's own drawing is hidden while the overlay draws.
  assert.equal(p.video.classList.contains("overlay-captions"), true);
  // A track not showing draws nothing.
  track.mode = "hidden";
  await show("Y");
  assert.deepEqual(drawn.at(-1), []);
});

test("captions cached before their track exists keep broadcast timing", async () => {
  const p = await player();
  const [cue] = p.engines[0].config.cueHandler.newCue(
    null,
    2,
    3,
    captionScreen("HELLO"),
  );
  assert.deepEqual([cue.startTime, cue.endTime, cue.text], [2, 3, "HELLO"]);
});
