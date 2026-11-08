const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const { createContext, runInContext } = require('node:vm');

function renderProgram(program) {
  const context = createContext({ program, window: {} });
  runInContext(readFileSync(join(__dirname, '../../netv/static/js/virtual-guide.js'), 'utf8'), context);
  return runInContext(`
    VirtualGuide.prototype.renderRow.call({
      currentRowHeight: 64, logoUrlFilter: value => value,
    }, {
      channel: { stream_id: '1', name: 'Stream' },
      programs: [program], programs_mobile: [program],
    }, 0)
  `, context);
}

test('virtual guide disables unavailable past listings on desktop and mobile', () => {
  const html = renderProgram({
    title: 'Earlier', start_timestamp: 100, end_timestamp: 200,
    catchup: false, unavailable: true,
  });
  const links = [...html.matchAll(/<a\s+([^>]+)>([\s\S]*?)<\/a>/g)]
    .filter(([, attributes]) => attributes.includes('title="Earlier'));
  assert.equal(links.length, 2);
  for (const [, attributes] of links) {
    assert.match(attributes, /aria-disabled="true"/);
    assert.match(attributes, /tabindex="-1"/);
    assert.match(attributes, /Not available in the upstream archive/);
    assert.doesNotMatch(attributes, /href=|data-nav=|focusable/);
  }
});

test('virtual guide retains archive URLs and current or future live links', () => {
  for (const catchup of [true, false]) {
    const html = renderProgram({
      title: 'Program', start_timestamp: 100, end_timestamp: 200,
      catchup, unavailable: false,
    });
    const expected = catchup ? '/play/live/1?start=100' : '/play/live/1';
    const links = [...html.matchAll(/<a\s+([^>]+)>([\s\S]*?)<\/a>/g)]
      .filter(([, attributes]) => attributes.includes('title="Program'));
    assert.equal(links.length, 2);
    for (const [, attributes] of links) {
      assert.ok(attributes.includes(`href="${expected}"`));
      assert.match(attributes, /data-nav="epg"/);
      assert.doesNotMatch(attributes, /aria-disabled/);
    }
  }
});
