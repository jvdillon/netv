// The guide pretunes the channel focus rests on (pretune.js).
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

function guide() {
  const listeners = { focusin: [], focusout: [] };
  const root = {
    addEventListener: (kind, fn) => listeners[kind].push(fn),
    contains: (el) => el?.inGuide === true,
  };
  const cell = (href) => ({
    inGuide: true,
    getAttribute: (name) => (name === "href" ? href : null),
  });
  const elsewhere = { inGuide: false, getAttribute: () => null };
  const requests = [];
  let now = 0;
  const timers = [];
  const context = createContext({
    window: {},
    fetch: async (url, options = {}) => {
      requests.push(`${options.method || "GET"} ${url}`);
      return { ok: true, json: async () => ({ pretuned: true }) };
    },
    setTimeout: (fn, ms) => timers.push({ fn, at: now + ms }) - 1,
    clearTimeout: (id) => {
      if (timers[id]) timers[id].fn = null;
    },
  });
  runInContext(
    readFileSync(join(__dirname, "../../netv/static/js/pretune.js"), "utf8"),
    context,
  );
  context.window.netvPretune(root);
  return {
    cell,
    elsewhere,
    requests,
    focus(target, from = null) {
      if (from)
        for (const fn of listeners.focusout)
          fn({ target: from, relatedTarget: target });
      for (const fn of listeners.focusin) fn({ target });
    },
    leave(from) {
      for (const fn of listeners.focusout)
        fn({ target: from, relatedTarget: elsewhere });
    },
    async advance(ms) {
      now += ms;
      for (const timer of timers)
        if (timer.fn && timer.at <= now) {
          const fn = timer.fn;
          timer.fn = null;
          await fn();
        }
      await new Promise((resolve) => setImmediate(resolve));
    },
  };
}

test("a channel is pretuned once focus has rested on it 600 ms", async () => {
  const g = guide();
  g.focus(g.cell("/play/aaaaaaaaaa"));
  await g.advance(599);
  assert.deepEqual(g.requests, []);
  await g.advance(1);
  assert.deepEqual(g.requests, ["POST /api/pretune/aaaaaaaaaa"]);
});

test("arrowing past channels pretunes only the one focus stops on", async () => {
  const g = guide();
  const [a, b, c] = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"].map((l) =>
    g.cell(`/play/${l}`),
  );
  g.focus(a);
  await g.advance(300);
  g.focus(b, a);
  await g.advance(300);
  g.focus(c, b);
  await g.advance(600);
  assert.deepEqual(g.requests, ["POST /api/pretune/cccccccccc"]);
});

test("moving along one channel's programmes keeps its rest going", async () => {
  const g = guide();
  const now = g.cell("/play/aaaaaaaaaa");
  g.focus(now);
  await g.advance(300);
  g.focus(g.cell("/play/aaaaaaaaaa"), now);
  await g.advance(300);
  assert.deepEqual(g.requests, ["POST /api/pretune/aaaaaaaaaa"]);
});

test("another programme of the same channel asks nothing more", async () => {
  const g = guide();
  const now = g.cell("/play/aaaaaaaaaa");
  g.focus(now);
  await g.advance(600);
  g.focus(g.cell("/play/aaaaaaaaaa"), now);
  await g.advance(600);
  assert.deepEqual(g.requests, ["POST /api/pretune/aaaaaaaaaa"]);
});

test("a past programme, played from the archive, is never pretuned", async () => {
  const g = guide();
  g.focus(g.cell("/play/aaaaaaaaaa?start=1700000000"));
  await g.advance(1000);
  assert.deepEqual(g.requests, []);
});

test("leaving the guide releases its pretune, and only one it holds", async () => {
  const g = guide();
  const a = g.cell("/play/aaaaaaaaaa");
  // Left before the dwell: nothing was tuned, so nothing is released.
  g.focus(a);
  await g.advance(100);
  g.leave(a);
  await g.advance(1000);
  assert.deepEqual(g.requests, []);
  g.focus(a);
  await g.advance(600);
  g.leave(a);
  await g.advance(0);
  assert.deepEqual(g.requests, ["POST /api/pretune/aaaaaaaaaa", "DELETE /api/pretune"]);
  // Coming back tunes it again.
  g.focus(a);
  await g.advance(600);
  assert.equal(g.requests.at(-1), "POST /api/pretune/aaaaaaaaaa");
});
