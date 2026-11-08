const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const script = readFileSync(join(__dirname, '../../netv/static/js/player-quality.js'), 'utf8');

function player(width = 0, height = 0) {
  const listeners = new Map();
  const classes = new Set(['hidden']);
  const attributes = new Map();
  const badge = {
    textContent: '',
    classList: {
      add: name => classes.add(name),
      remove: name => classes.delete(name),
    },
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: name => attributes.delete(name),
  };
  const video = {
    videoWidth: width,
    videoHeight: height,
    addEventListener: (name, callback) => listeners.set(name, callback),
  };
  runInNewContext(script, {
    document: { getElementById: id => id === 'video' ? video : badge },
  });
  return { video, badge, classes, attributes, fire: event => listeners.get(event)() };
}

test('resolution labels match Apple, including letterboxed video', () => {
  for (const [width, height, label] of [
    [3840, 2160, '4K'], [3840, 1600, '4K'], [2560, 1440, '1440p'],
    [1920, 1080, '1080p'], [1920, 800, '1080p'], [1280, 720, '720p'],
    [720, 576, '576p'], [720, 480, '480p'], [640, 360, 'SD'],
  ]) {
    const { badge, classes, attributes } = player(width, height);
    assert.equal(badge.textContent, label);
    assert.equal(classes.has('hidden'), false);
    assert.equal(attributes.get('aria-label'), `Video resolution: ${label}`);
  }
});

test('uses decoded dimensions and follows resolution changes', () => {
  const p = player();
  assert.equal(p.classes.has('hidden'), true);
  p.video.videoWidth = 1280;
  p.video.videoHeight = 720;
  p.fire('loadedmetadata');
  assert.equal(p.badge.textContent, '720p');
  p.video.videoWidth = 3840;
  p.video.videoHeight = 2160;
  p.fire('resize');
  assert.equal(p.badge.textContent, '4K');
  p.video.videoWidth = 0;
  p.fire('resize');
  assert.equal(p.badge.textContent, '');
  assert.equal(p.classes.has('hidden'), true);
});

test('classification thresholds match the Apple badge', () => {
  for (const [height, label] of [
    [2000, '4K'], [1999, '1440p'], [1300, '1440p'], [1299, '1080p'],
    [900, '1080p'], [899, '720p'], [650, '720p'], [649, '576p'],
    [520, '576p'], [519, '480p'], [400, '480p'], [399, 'SD'],
  ]) {
    assert.equal(player(1, height).badge.textContent, label);
  }
  assert.equal(player(1920, 0).classes.has('hidden'), true);
  assert.equal(player(0, 1080).classes.has('hidden'), true);
});

test('clears stale quality on reload, detach, and error; restores on playback', () => {
  for (const event of ['loadstart', 'emptied', 'error']) {
    const p = player(1920, 1080);
    p.fire(event);
    assert.equal(p.classes.has('hidden'), true);
    assert.equal(p.badge.textContent, '');
    assert.equal(p.attributes.has('aria-label'), false);
    p.fire('playing');
    assert.equal(p.badge.textContent, '1080p');
  }
});

test('can load on pages without the player', () => {
  runInNewContext(script, { document: { getElementById: () => null } });
});
