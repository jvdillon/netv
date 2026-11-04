const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const script = readFileSync(join(__dirname, '../static/js/cast.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function element() {
  const listeners = new Map();
  const classes = new Set(['hidden']);
  return {
    disabled: false, value: '', textContent: '', children: [],
    classList: {
      add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name),
      toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
    },
    addEventListener(name, handler) { listeners.set(name, handler); },
    emit(name, event = {}) { return listeners.get(name)?.(event); },
    appendChild(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    showModal() { this.open = true; }, close() { this.open = false; },
  };
}

async function setup({ origin = 'https://netv.test', isVod = true, resumed = false,
  cancel = false, loadFailure = false, prepare, initialServer = false,
  resumedUrl = `${origin}/transcode/existing/stream.m3u8` } = {}) {
  const elements = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  const calls = [], order = [], timers = [];
  let serverStarted = 0, browserStarted = 0, reloads = 0, loaded = resumed;
  let serverStatus = initialServer
    ? { active: true, name: 'Server TV', state: 'PLAYING', volume: 0.5 }
    : { active: false };
  let currentSession;
  let sessionListener;
  const mediaSession = {
    playerState: 'PLAYING', media: { contentId: resumedUrl },
    play(request, resolve) { order.push('play'); this.playerState = 'PLAYING'; resolve(); },
    pause(request, resolve) { order.push('pause'); this.playerState = 'PAUSED'; resolve(); },
  };
  const session = {
    getMediaSession: () => loaded ? mediaSession : null,
    getCastDevice: () => ({ friendlyName: 'Nearby TV' }),
    getVolume: () => 0.4,
    async setVolume(volume) { order.push(['volume', volume]); },
    async loadMedia(request) {
      order.push('load'); calls.push({ request });
      if (loadFailure) throw new Error('Receiver could not load media');
      loaded = true;
    },
  };
  if (resumed) currentSession = session;
  const castContext = {
    setOptions(options) { this.options = options; },
    addEventListener(type, handler) { sessionListener = handler; },
    getCurrentSession: () => currentSession,
    requestSession() {
      order.push('picker');
      if (cancel) return Promise.reject(new Error('cancel'));
      currentSession = session;
      return Promise.resolve();
    },
    endCurrentSession(stop) {
      order.push(['end', stop]); currentSession = null; loaded = false;
      sessionListener({ sessionState: 'ended' });
    },
  };
  const window = Object.assign(element(), {
    location: Object.assign(new URL(origin), { reload: () => reloads++ }),
  });
  const document = {
    getElementById: id => el(id.replace(/^cast-/, '')),
    createElement: element,
    head: element(),
  };
  document.head.appendChild = child => {
    assert.equal(typeof window.__onGCastApiAvailable, 'function');
    document.head.children.push(child);
  };
  runInNewContext(script, {
    window, document, AbortSignal, URL,
    console: { warn() {} },
    setTimeout: fn => { timers.push(fn); return timers.length; }, clearTimeout() {},
    fetch: async (url, options) => {
      const body = options.body && JSON.parse(options.body);
      calls.push({ url, body });
      let data = serverStatus;
      if (url.endsWith('/devices')) data = { devices: [{ name: 'Server TV', host: '192.168.2.10' }] };
      if (url.endsWith('/start')) data = serverStatus = {
        active: true, name: 'Server TV', state: 'PLAYING', volume: 0.5,
      };
      if (url.endsWith('/control') && body.action === 'stop') data = serverStatus = { active: false };
      return { ok: true, json: async () => data };
    },
  });
  const controller = window.NetvCast.setup({
    serverAddress: origin,
    getMedia: async () => ({ session_id: 'server-stream', current_time: 42 }),
    onStarted: () => serverStarted++,
    getBrowserMedia: async () => {
      order.push('prepare');
      if (prepare) await prepare;
      return { url: `${origin}/transcode/browser-stream/stream.m3u8`, title: 'A title', isVod, current_time: 42 };
    },
    onBrowserStarted: () => { browserStarted++; order.push('browser-started'); },
  });
  await controller.ready;
  function initialize() {
    window.cast = { framework: {
      CastContext: { getInstance: () => castContext },
      CastContextEventType: { SESSION_STATE_CHANGED: 'session' },
      SessionState: { SESSION_ENDED: 'ended' },
    } };
    window.chrome = { cast: {
      AutoJoinPolicy: { ORIGIN_SCOPED: 'origin' },
      media: {
        DEFAULT_MEDIA_RECEIVER_APP_ID: 'default-receiver',
        StreamType: { BUFFERED: 'buffered', LIVE: 'live' },
        MediaInfo: function(url, type) { this.contentId = url; this.contentType = type; },
        GenericMediaMetadata: function() {},
        LoadRequest: function(media) { this.media = media; },
      },
    } };
    window.__onGCastApiAvailable(true);
  }
  return {
    el, window, document, controller, calls, order, castContext, initialize,
    poll: () => timers.at(-1)(),
    ended: () => { currentSession = null; sessionListener({ sessionState: 'ended' }); },
    get serverStarted() { return serverStarted; },
    get browserStarted() { return browserStarted; },
    get reloads() { return reloads; },
  };
}

test('SDK callback precedes dynamic script and enables nearby picker after initialization', async () => {
  const p = await setup();
  assert.equal(p.document.head.children[0].src,
    'https://www.gstatic.com/cv/js/sender/v1/cast_sender.js?loadCastFramework=1');
  assert.equal(p.el('browser').disabled, true);
  p.initialize();
  assert.equal(p.el('browser').disabled, false);
  assert.equal(p.castContext.options.receiverApplicationId, 'default-receiver');
  assert.equal(p.castContext.options.autoJoinPolicy, 'origin');
});

test('nearby picker runs synchronously before preparing media; successful VOD uses browser callback only', async () => {
  let release;
  const p = await setup({ prepare: new Promise(resolve => { release = resolve; }) });
  p.initialize();
  const click = p.el('browser').emit('click');
  assert.deepEqual(p.order, ['picker']);
  await flush();
  assert.deepEqual(p.order, ['picker', 'prepare']);
  assert.equal(p.el('browser').disabled, true);
  await p.el('form').emit('submit', { preventDefault() {} });
  assert.equal(p.calls.some(call => call.url?.endsWith('/start')), false);
  release();
  await click;
  const request = p.calls.find(call => call.request).request;
  assert.equal(request.media.contentId, 'https://netv.test/transcode/browser-stream/stream.m3u8');
  assert.equal(request.media.contentType, 'application/x-mpegURL');
  assert.equal(request.media.metadata.title, 'A title');
  assert.equal(request.media.streamType, 'buffered');
  assert.equal(request.media.hlsSegmentFormat, 'ts');
  assert.equal(request.media.hlsVideoSegmentFormat, 'mpeg2_ts');
  assert.equal(request.currentTime, 42);
  assert.equal(p.browserStarted, 1);
  assert.equal(p.serverStarted, 0);
  assert.equal(p.controller.browserActive, true);
  await p.poll();
  assert.equal(p.controller.active, true);
  assert.equal(p.el('picker').classList.contains('hidden'), true);
  assert.equal(p.el('note').textContent, 'Keep neTV reachable while casting.');
});

test('live browser media does not seek and remote controls target the browser session', async () => {
  const p = await setup({ isVod: false });
  p.initialize();
  await p.el('browser').emit('click');
  const request = p.calls.find(call => call.request).request;
  assert.equal(request.media.streamType, 'live');
  assert.equal(request.currentTime, undefined);
  await p.el('pause').emit('click');
  assert.match(p.el('state').textContent, /paused/);
  await p.el('play').emit('click');
  p.el('volume').value = '0.7';
  await p.el('volume').emit('change');
  assert.ok(p.order.some(item => Array.isArray(item) && item[0] === 'volume' && item[1] === 0.7));
  await p.el('local').emit('click');
  assert.equal(p.controller.browserActive, false);
  assert.equal(p.controller.active, false);
  assert.equal(p.reloads, 1);
  assert.ok(p.order.some(item => Array.isArray(item) && item[0] === 'end' && item[1]));
  assert.equal(p.calls.some(call => call.url?.endsWith('/control')), false);
});

test('SDK unavailable leaves server discovery, media preparation and controls working', async () => {
  const p = await setup();
  p.document.head.children[0].onerror();
  assert.equal(p.el('browser').disabled, true);
  assert.match(p.el('browser-status').textContent, /unavailable/);
  await p.controller.open();
  assert.match(p.el('devices').children[0].textContent, /Server TV/);
  p.el('ip').value = '192.168.2.10';
  await p.el('form').emit('submit', { preventDefault() {} });
  await flush();
  const start = p.calls.find(call => call.url?.endsWith('/start'));
  assert.deepEqual(start.body, {
    session_id: 'server-stream', current_time: 42, host: '192.168.2.10', server_url: 'https://netv.test',
  });
  assert.equal(p.serverStarted, 1);
  assert.equal(p.browserStarted, 0);
  await p.el('pause').emit('click');
  assert.deepEqual(p.calls.at(-1).body, { action: 'pause' });
});

test('insecure origins explain fallback, but localhost can initialize', async () => {
  const p = await setup({ origin: 'http://netv.test' });
  assert.equal(p.document.head.children.length, 0);
  assert.match(p.el('browser-status').textContent, /HTTPS or localhost/);
  const local = await setup({ origin: 'http://localhost:8000' });
  local.initialize();
  assert.equal(local.el('browser').disabled, false);
});

for (const failure of ['cancel', 'loadFailure']) {
  test(`${failure} recovers controls without invoking either handoff callback`, async () => {
    const p = await setup({ [failure]: true });
    p.initialize();
    await p.el('browser').emit('click');
    assert.equal(p.controller.active, false);
    assert.equal(p.controller.browserActive, false);
    assert.equal(p.controller.busy, false);
    assert.equal(p.el('browser').disabled, false);
    assert.equal(p.el('start').disabled, false);
    assert.equal(p.browserStarted, 0);
    assert.equal(p.serverStarted, 0);
    assert.equal(p.el('error').classList.contains('hidden'), false);
    if (failure === 'cancel') assert.equal(p.order.includes('prepare'), false);
    else assert.ok(p.order.some(item => Array.isArray(item) && item[0] === 'end'));
    await p.el('form').emit('submit', { preventDefault() {} });
    await flush();
    assert.equal(p.serverStarted, 1);
  });
}

test('resumed sessions use browser state and survive inactive server polls', async () => {
  const p = await setup({ resumed: true });
  p.initialize();
  assert.equal(p.controller.browserActive, true);
  assert.equal(p.browserStarted, 1);
  assert.equal(p.order.includes('picker'), false);
  assert.equal(p.order.includes('prepare'), false);
  await p.poll();
  assert.equal(p.controller.active, true);
  assert.equal(p.serverStarted, 0);
  p.ended();
  assert.equal(p.controller.active, false);
  assert.equal(p.el('browser').disabled, false);
});

test('unrelated default-receiver media is not resumed as neTV casting', async () => {
  for (const resumedUrl of ['https://other.test/transcode/id/stream.m3u8', 'https://netv.test/movie.mp4']) {
    const p = await setup({ resumed: true, resumedUrl });
    p.initialize();
    assert.equal(p.controller.browserActive, false);
    assert.equal(p.browserStarted, 0);
    assert.equal(p.el('browser').disabled, false);
  }
});

test('browser stop restores the picker without reloading', async () => {
  const p = await setup();
  p.initialize();
  await p.el('browser').emit('click');
  await p.el('stop').emit('click');
  assert.equal(p.controller.active, false);
  assert.equal(p.el('picker').classList.contains('hidden'), false);
  assert.equal(p.el('browser').disabled, false);
  assert.equal(p.reloads, 0);
});

test('an active server session blocks nearby starts', async () => {
  const p = await setup({ initialServer: true });
  p.initialize();
  assert.equal(p.el('browser').disabled, true);
  await p.el('browser').emit('click');
  assert.equal(p.order.includes('picker'), false);
  assert.equal(p.controller.browserActive, false);
  assert.equal(p.serverStarted, 1);
});
