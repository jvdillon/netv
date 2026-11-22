const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

function renderProgram(program) {
  const context = createContext({ program, window: {} });
  runInContext(
    readFileSync(join(__dirname, "../../netv/static/js/virtual-guide.js"), "utf8"),
    context,
  );
  return runInContext(
    `
    VirtualGuide.prototype.renderRow.call({
      currentRowHeight: 64, logoUrlFilter: value => value,
    }, {
      channel: { stream_id: '1', name: 'Stream' },
      programs: [program], programs_mobile: [program],
    }, 0)
  `,
    context,
  );
}

// A fake of the guide's rows: render() replaces them, as innerHTML does.
function guideWithFocus(focused) {
  const calls = [];
  const document = { activeElement: focused };
  const cell = (row, col) => {
    const el = {
      dataset: { nav: "epg", row: String(row), col: String(col) },
      offsetParent: {},
      scrollIntoView: (options) =>
        calls.push(JSON.parse(JSON.stringify({ row, col, shown: options }))),
      focus: (options) => {
        document.activeElement = el;
        // As plain data: options come from the script's own realm.
        calls.push(JSON.parse(JSON.stringify({ row, col, options })));
      },
    };
    return el;
  };
  let cells = [focused];
  const content = {
    style: {},
    contains: (el) => cells.includes(el),
    set innerHTML(_) {
      cells = [cell(5, 2), cell(5, -1), cell(9, 0), cell(9, -1)];
      document.activeElement = { dataset: {} };
    },
    querySelectorAll: (selector) => {
      const [, row, col] = selector.match(/data-row="(\d+)"\]\[data-col="(-?\d+)"/);
      return cells.filter((c) => c.dataset.row === row && c.dataset.col === col);
    },
  };
  const context = createContext({ window: {}, document });
  runInContext(
    readFileSync(join(__dirname, "../../netv/static/js/virtual-guide.js"), "utf8"),
    context,
  );
  const guide = runInContext("Object.create(VirtualGuide.prototype)", context);
  Object.assign(guide, {
    content,
    renderedRange: { start: 5, end: 10 },
    sections: new Map(),
    cache: new Map(),
    pendingFocus: null,
    rowTop: () => 0,
    renderRow: () => "",
    renderPlaceholder: () => "",
  });
  return { guide, calls };
}

test("virtual guide keeps keyboard focus when it re-renders its rows", () => {
  const { guide, calls } = guideWithFocus({
    dataset: { nav: "epg", row: "5", col: "2" },
  });
  guide.render();
  // The same programme, without scrolling the page to it.
  assert.deepEqual(calls, [{ row: 5, col: 2, options: { preventScroll: true } }]);
});

test("virtual guide focuses a row a key asked for once it arrives", () => {
  const { guide, calls } = guideWithFocus({
    dataset: { nav: "epg", row: "5", col: "2" },
  });
  guide.pendingFocus = { row: 9, col: 0 };
  guide.render();
  // The row asked for is scrolled into the window too, as the last row must be.
  assert.deepEqual(calls, [
    { row: 9, col: 0, options: { preventScroll: true } },
    { row: 9, col: 0, shown: { block: "nearest", inline: "nearest" } },
  ]);
  assert.equal(guide.pendingFocus, null);
  // A row not yet fetched has no cells: focus stays where it was, unscrolled,
  // and the request waits for the next render.
  guide.pendingFocus = { row: 12, col: -1 };
  guide.render();
  assert.deepEqual(calls.slice(2), [
    { row: 9, col: 0, options: { preventScroll: true } },
  ]);
  assert.deepEqual(guide.pendingFocus, { row: 12, col: -1 });
});

test("virtual guide falls back to the channel when the column is gone", () => {
  const { guide, calls } = guideWithFocus({
    dataset: { nav: "epg", row: "9", col: "3" },
  });
  guide.render();
  assert.deepEqual(calls, [{ row: 9, col: -1, options: { preventScroll: true } }]);
});

test("virtual guide scrolls far enough to show its last row under the time header", () => {
  const context = createContext({ window: { innerWidth: 1600 }, document: {} });
  runInContext(
    readFileSync(join(__dirname, "../../netv/static/js/virtual-guide.js"), "utf8"),
    context,
  );
  const guide = runInContext("Object.create(VirtualGuide.prototype)", context);
  Object.assign(guide, {
    totalRows: 10,
    rowHeight: 64,
    sectionHeight: 32,
    sectionStarts: [0, 5],
    isMobile: false,
    // The time header sits above the rows inside the scrolling viewport.
    content: { offsetTop: 41 },
  });
  assert.equal(guide.scrollableHeight, 10 * 64 + 2 * 32 + 41);
});

test("virtual guide disables unavailable past listings on desktop and mobile", () => {
  const html = renderProgram({
    title: "Earlier",
    start_timestamp: 100,
    end_timestamp: 200,
    catchup: false,
    unavailable: true,
  });
  const links = [...html.matchAll(/<a\s+([^>]+)>([\s\S]*?)<\/a>/g)].filter(
    ([, attributes]) => attributes.includes('title="Earlier'),
  );
  assert.equal(links.length, 2);
  for (const [, attributes] of links) {
    assert.match(attributes, /aria-disabled="true"/);
    assert.match(attributes, /tabindex="-1"/);
    assert.match(attributes, /Not available in the upstream archive/);
    assert.doesNotMatch(attributes, /href=|data-nav=|focusable/);
  }
});

test("virtual guide retains archive URLs and current or future live links", () => {
  for (const catchup of [true, false]) {
    const html = renderProgram({
      title: "Program",
      start_timestamp: 100,
      end_timestamp: 200,
      catchup,
      unavailable: false,
    });
    const expected = catchup ? "/play/live/1?start=100" : "/play/live/1";
    const links = [...html.matchAll(/<a\s+([^>]+)>([\s\S]*?)<\/a>/g)].filter(
      ([, attributes]) => attributes.includes('title="Program'),
    );
    assert.equal(links.length, 2);
    for (const [, attributes] of links) {
      assert.ok(attributes.includes(`href="${expected}"`));
      assert.match(attributes, /data-nav="epg"/);
      assert.doesNotMatch(attributes, /aria-disabled/);
    }
  }
});
