const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const script = readFileSync(join(__dirname, '../static/js/live-playback.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
const ok = data => ({ ok: true, json: async () => data });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

function runtime(fetch) {
  const timers = new Map();
  const warnings = [];
  const beacons = [];
  let clock = 1000;
  let timerId = 0;
  const window = { location: { href: 'http://netv.test/play/live/1', origin: 'http://netv.test' } };
  const Events = { FRAG_LOADED: 'fragment', MANIFEST_PARSED: 'manifest', LEVEL_SWITCHED: 'level' };
  runInNewContext(script, {
    window, fetch, URL, Blob, AbortController, Hls: { Events },
    performance: { now: () => clock },
    console: { warn: (...args) => warnings.push(args) },
    navigator: { sendBeacon: url => { beacons.push(url); return true; } },
    setTimeout: (fn, ms) => { timers.set(++timerId, { fn, ms }); return timerId; },
    clearTimeout: id => timers.delete(id),
  });
  return {
    ...window.NetvPlayback, timers, warnings, beacons,
    advance: ms => { clock += ms; },
    async tick() {
      const [id, timer] = timers.entries().next().value;
      timers.delete(id);
      assert.equal(timer.ms, 2000);
      await timer.fn();
    },
  };
}

test('rapid live starts release the earlier session before opening another', async () => {
  const first = deferred();
  const calls = [];
  const env = runtime(async (url, options) => {
    calls.push([url, options?.method || 'GET']);
    if (url === '/start/a') return first.promise;
    if (url === '/start/b') return ok({ session_id: 'b', playlist: '/b' });
    return ok({});
  });
  const session = new env.TranscodeSession(false);
  const a = session.start('/start/a');
  await flush();
  const b = session.start('/start/b');
  first.resolve(ok({ session_id: 'a', playlist: '/a' }));
  assert.equal(await a, null);
  assert.equal((await b).session_id, 'b');
  assert.deepEqual(calls, [
    ['/start/a', 'GET'], ['/transcode/a?force=true', 'DELETE'], ['/start/b', 'GET'],
  ]);
});

test('failed stop blocks replacement but can be retried', async () => {
  const calls = [];
  let failStop = true;
  const env = runtime(async (url, options) => {
    calls.push(url);
    if (options?.method === 'DELETE') {
      return failStop ? { ok: false, status: 503 } : ok({});
    }
    return ok({ session_id: url.endsWith('/a') ? 'a' : 'c', playlist: '/playlist' });
  });
  const session = new env.TranscodeSession(false);
  await session.start('/start/a');
  await assert.rejects(session.start('/start/b'), /stop failed: 503/);
  assert.equal(session.sessionId, 'a');
  assert.equal(calls.includes('/start/b'), false);
  failStop = false;
  assert.equal((await session.start('/start/c')).session_id, 'c');
});

test('page close during startup releases the eventual live session', async () => {
  const first = deferred();
  const calls = [];
  const env = runtime(async (url, options) => {
    calls.push([url, options]);
    return url === '/start' ? first.promise : ok({});
  });
  const session = new env.TranscodeSession(false);
  const pending = session.start('/start');
  await flush();
  session.close();
  first.resolve(ok({ session_id: 'late', playlist: '/playlist' }));
  assert.equal(await pending, null);
  assert.equal(session.sessionId, null);
  assert.equal(calls[1][0], '/transcode/late?force=true');
  assert.equal(calls[1][1].keepalive, true);
});

test('VOD retains cached-stop semantics; live page close forces release', async () => {
  for (const isVod of [false, true]) {
    const env = runtime(async () => ok({ session_id: 'active', playlist: '/playlist' }));
    const session = new env.TranscodeSession(isVod);
    await session.start('/start');
    session.close();
    assert.deepEqual(env.beacons, [`/transcode/active/stop?force=${!isVod}`]);
  }
});

test('playback buffer hints retain compatibility defaults and bounded latency', () => {
  const env = runtime(async () => ok({}));
  assert.equal(env.playbackBufferSeconds(undefined), 12);
  assert.equal(env.playbackBufferSeconds(8), 12);
  assert.equal(env.playbackBufferSeconds('30'), 30);
  assert.equal(env.playbackBufferSeconds(90), 60);
});

test('casting detaches the session from browser stop and page-close cleanup', async () => {
  const calls = [];
  const env = runtime(async (url, options) => {
    calls.push([url, options?.method || 'GET']);
    return ok({ session_id: 'cast', playlist: '/playlist' });
  });
  const session = new env.TranscodeSession(false);
  await session.start('/start');
  session.detach();
  await session.stop();
  session.close();
  assert.deepEqual(calls, [['/start', 'GET']]);
  assert.deepEqual(env.beacons, []);
});

function video() {
  return {
    currentTime: 10, paused: false, ended: false, readyState: 4,
    buffered: { length: 1, start: () => 0, end: () => 20 },
  };
}

function hls() {
  const listeners = new Map();
  const switches = [];
  return {
    levels: [
      { url: ['http://netv.test/transcode/shared/low.m3u8'] },
      { uri: 'http://netv.test/transcode/shared/high.m3u8' },
    ],
    switches,
    on: (event, fn) => listeners.set(event, fn),
    off: event => listeners.delete(event),
    fire: (event, data) => listeners.get(event)?.(event, data),
    set nextLevel(level) { switches.push(level); },
    loadSource: () => assert.fail('A rendition switch must not reload the source'),
    detachMedia: () => assert.fail('A rendition switch must preserve the media element'),
  };
}

test('backend health controls upgrades and downgrades within one Hls instance', async () => {
  const reports = [];
  const shown = [];
  let rendition = 'low';
  const env = runtime(async (url, options) => {
    assert.equal(url, '/transcode/shared/health');
    reports.push(JSON.parse(options.body));
    return ok({ playlist: `/transcode/shared/${rendition}.m3u8` });
  });
  const media = video();
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: media, hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: value => shown.push(value),
  });
  await flush();
  assert.equal(engine.startLevel, 0);
  assert.equal(engine.autoLevelCapping, 0);
  assert.deepEqual(engine.switches, [0]);
  engine.fire('fragment', {
    frag: { type: 'main', duration: 2, stats: { loaded: 250000, loading: { start: 10, end: 30 } } },
  });
  rendition = 'high';
  await env.tick();
  assert.equal(reports[1].observed_bitrate, 100000000);
  assert.equal(reports[1].required_bitrate, 1000000);
  assert.equal(reports[1].buffer_seconds, 10);
  assert.deepEqual(engine.switches, [0, 1]);
  engine.fire('level', { level: 1 });
  assert.deepEqual(shown, ['/transcode/shared/high.m3u8']);
  await env.tick();
  assert.equal(reports[2].observed_bitrate, 0);
  assert.deepEqual(engine.switches, [0, 1]);
  rendition = 'low';
  await env.tick();
  assert.deepEqual(engine.switches, [0, 1, 0]);
  playback.destroy();
  assert.equal(env.timers.size, 0);
});

test('recovery generations reuse the existing Hls rendition levels', async () => {
  let generation = 1;
  const env = runtime(async () => ok({
    playlist: `/transcode/shared/low.m3u8?generation=${generation}`,
  }));
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: video(), hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  await flush();
  assert.deepEqual(engine.switches, [0]);
  generation = 2;
  await env.tick();
  assert.deepEqual(engine.switches, [0, 0]);
  playback.destroy();
});

test('pauses, stale transfers, and gaps do not create false throughput samples', async () => {
  const env = runtime(async () => ok({}));
  const media = video();
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: media, hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  await flush();
  engine.fire('fragment', {
    frag: { type: 'main', duration: 2, stats: { loaded: 1000, loading: { start: 0, end: 20 } } },
  });
  env.advance(11000);
  assert.equal(playback.sample().observed_bitrate, 0);
  media.buffered = { length: 1, start: () => 30, end: () => 40 };
  media.paused = true;
  media.readyState = 1;
  const sample = playback.sample();
  assert.equal(sample.buffer_seconds, 0);
  assert.equal(sample.waiting, false);
  assert.equal(sample.required_bitrate, 0);
  playback.destroy();
});

test('destroyed playback ignores late telemetry and cancels its timer', async () => {
  const response = deferred();
  let signal;
  const env = runtime(async (_url, options) => { signal = options.signal; return response.promise; });
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: video(), hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  playback.destroy();
  response.resolve(ok({ playlist: '/transcode/shared/high.m3u8' }));
  await flush();
  assert.equal(signal.aborted, true);
  assert.deepEqual(engine.switches, []);
  assert.equal(env.timers.size, 0);
});

test('telemetry failure leaves the current rendition and retries without overlap', async () => {
  let calls = 0;
  const env = runtime(async () => { calls++; return { ok: false, status: 503 }; });
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: video(), hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  await flush();
  assert.equal(calls, 1);
  assert.equal(env.timers.size, 1);
  assert.deepEqual(engine.switches, []);
  await env.tick();
  assert.equal(calls, 2);
  assert.equal(env.warnings.length, 1);
  playback.destroy();
});

test('native HLS sends heartbeats but cannot fabricate upgrade evidence', async () => {
  let report;
  const env = runtime(async (_url, options) => {
    report = JSON.parse(options.body);
    return ok({ playlist: '/transcode/shared/high.m3u8' });
  });
  const playback = new env.AdaptiveLivePlayback({
    video: video(), hls: null, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8',
    onPlaylist: () => assert.fail('Native playback must stay on its initial rendition'),
  });
  await flush();
  assert.equal(report.observed_bitrate, 0);
  assert.equal(env.timers.size, 1);
  playback.destroy();
});

test('paused playback does not apply a pending upgrade', async () => {
  const env = runtime(async () => ok({ playlist: '/transcode/shared/high.m3u8' }));
  const media = video();
  media.paused = true;
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: media, hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  await flush();
  assert.deepEqual(engine.switches, []);
  playback.destroy();
});

test('invalid cross-session rendition keeps playback unchanged and reports the error', async () => {
  const env = runtime(async () => ok({ playlist: '/transcode/other/high.m3u8' }));
  const engine = hls();
  const playback = new env.AdaptiveLivePlayback({
    video: video(), hls: engine, sessionId: 'shared',
    playlist: '/transcode/shared/low.m3u8', onPlaylist: () => {},
  });
  await flush();
  assert.deepEqual(engine.switches, []);
  assert.equal(env.warnings.length, 1);
  playback.destroy();
});
