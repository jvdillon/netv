// Keyboard navigation (app.js) on a small fake page: boxes, focus, and keys.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

// One compound selector, e.g. a[href]:not([disabled]), .grid, [data-nav="row"].
function matchesCompound(node, compound) {
  const not = compound.match(/:not\(\[(\w[\w-]*)\]\)/);
  if (not && node.attrs[not[1]] !== undefined) return false;
  const rest = compound.replace(/:not\([^)]*\)/g, "");
  const tag = rest.match(/^[a-z]+/);
  if (tag && node.tag !== tag[0]) return false;
  for (const [, name] of rest.matchAll(/\.([\w-]+)/g))
    if (!node.classes.has(name)) return false;
  for (const [, name, value] of rest.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
    if (node.attrs[name] === undefined) return false;
    if (value !== undefined && node.attrs[name] !== value) return false;
  }
  return true;
}

function matches(node, selector) {
  return selector.split(",").some((part) => {
    const steps = part.trim().split(/\s+/);
    if (!matchesCompound(node, steps.at(-1))) return false;
    let at = node.parent;
    for (const step of steps.slice(0, -1).reverse()) {
      while (at && !matchesCompound(at, step)) at = at.parent;
      if (!at) return false;
      at = at.parent;
    }
    return true;
  });
}

function page() {
  const listeners = { keydown: [], focusin: [], DOMContentLoaded: [] };
  const observers = [];
  const document = {
    readyState: "complete",
    activeElement: null,
    addEventListener: (kind, fn) => listeners[kind]?.push(fn),
    querySelectorAll: (selector) => document.body.querySelectorAll(selector),
    querySelector: (selector) => document.body.querySelector(selector),
  };
  // A node with its box (x, y, w, h); hidden nodes take no focus.
  function el(tag, attrs = {}, box = [0, 0, 10, 10], ...children) {
    const { class: classes = "", hidden = false, ...rest } = attrs;
    const node = {
      tag,
      attrs: rest,
      classes: new Set(classes.split(" ").filter(Boolean)),
      classList: null,
      hidden,
      parent: null,
      children: [],
      tagName: tag.toUpperCase(),
      get parentElement() {
        return this.parent;
      },
      get href() {
        return this.attrs.href;
      },
      get offsetWidth() {
        return box[2];
      },
      getBoundingClientRect: () => ({
        left: box[0],
        top: box[1],
        width: box[2],
        height: box[3],
        right: box[0] + box[2],
        bottom: box[1] + box[3],
      }),
      checkVisibility() {
        for (let at = this; at; at = at.parent) if (at.hidden) return false;
        return true;
      },
      get isConnected() {
        let at = this;
        while (at.parent) at = at.parent;
        return at === document.body;
      },
      matches(selector) {
        return matches(this, selector);
      },
      closest(selector) {
        for (let at = this; at; at = at.parent) if (matches(at, selector)) return at;
        return null;
      },
      contains(other) {
        for (let at = other; at; at = at.parent) if (at === this) return true;
        return false;
      },
      querySelectorAll(selector) {
        const found = [];
        const walk = (n) => {
          for (const c of n.children) {
            if (matches(c, selector)) found.push(c);
            walk(c);
          }
        };
        walk(this);
        return found;
      },
      querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
      },
      append(...nodes) {
        for (const n of nodes) {
          n.parent = this;
          this.children.push(n);
        }
        for (const o of observers) o();
      },
      focus() {
        document.activeElement = this;
        for (const fn of listeners.focusin) fn({ target: this });
      },
      scrollIntoView() {},
      click() {
        this.clicked = (this.clicked || 0) + 1;
      },
    };
    node.classList = { contains: (name) => node.classes.has(name) };
    for (const c of children) {
      c.parent = node;
      node.children.push(c);
    }
    return node;
  }
  const window = { location: { href: "", pathname: "/page" }, open() {} };
  const history = {
    backs: 0,
    back() {
      this.backs++;
    },
  };
  const context = createContext({
    window,
    document,
    history,
    location: window.location,
    setTimeout: (fn) => fn(),
    MutationObserver: class {
      constructor(fn) {
        this.fn = fn;
      }
      observe() {
        observers.push(() => this.fn([]));
      }
    },
    fetch: async () => ({ ok: false }),
    performance: { getEntriesByType: () => [] },
  });
  window.addEventListener = () => {};
  return {
    el,
    document,
    window,
    history,
    // Builds the page from `body`, then loads app.js as the browser would.
    load(body) {
      document.body = body;
      runInContext(
        readFileSync(join(__dirname, "../../netv/static/js/app.js"), "utf8"),
        context,
      );
    },
    press(key, extra = {}) {
      const event = {
        key,
        target: document.activeElement || document.body,
        altKey: false,
        ctrlKey: false,
        metaKey: false,
        defaultPrevented: false,
        preventDefault() {
          this.defaultPrevented = true;
        },
        ...extra,
      };
      for (const fn of listeners.keydown) fn(event);
      return event;
    },
  };
}

// A sidebar of three links (Movies active) and a main region the test fills.
function withSidebar(p, ...main) {
  const { el } = p;
  const links = ["Live", "Movies", "Search"].map((name, i) =>
    el(
      "a",
      {
        href: `/${name}`,
        class: `nav-link${name === "Movies" ? " active" : ""}`,
        name,
      },
      [0, 100 * i, 40, 40],
    ),
  );
  const body = el(
    "body",
    {},
    [0, 0, 1600, 900],
    el("nav", {}, [0, 0, 40, 900], ...links),
    el("main", {}, [60, 0, 1540, 900], ...main),
  );
  return { body, links };
}

test("Down enters a row of small far-left items below, not a central item further down", () => {
  const p = page();
  const { el } = p;
  // The search page at 1600 wide: the box; under it small checkboxes at its left end;
  // then a field a line further down, but nearer the box's center than any checkbox.
  const box = el("input", { type: "text" }, [80, 100, 700, 40]);
  const boxes = ["live", "vod", "series"].map((name, i) =>
    el("input", { type: "checkbox", name }, [80 + 90 * i, 160, 16, 16]),
  );
  const row = el("div", { "data-nav": "row" }, [80, 156, 300, 24], ...boxes);
  const far = el("input", { type: "number", name: "min_imdb" }, [290, 220, 60, 30]);
  const { body } = withSidebar(p, box, row, far);
  p.load(body);
  box.focus();
  p.press("ArrowDown");
  assert.equal(p.document.activeElement.attrs.name, "series");
  // Left and Right walk the row in order; the first box is reached.
  p.press("ArrowLeft");
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement.attrs.name, "live");
  p.press("ArrowRight");
  assert.equal(p.document.activeElement.attrs.name, "vod");
});

test("a text field entered sideways is left by the opposite key at once", () => {
  const p = page();
  const { el } = p;
  // Settings: a folder field between a link and its Delete button.
  const before = el("a", { href: "/x", name: "before" }, [80, 100, 40, 30]);
  const field = el("input", { type: "text", name: "dir" }, [140, 100, 300, 30]);
  Object.assign(field, {
    type: "text",
    value: "/home/user/data",
    selectionStart: 0,
    selectionEnd: 0,
    setSelectionRange(start, end) {
      Object.assign(this, { selectionStart: start, selectionEnd: end });
    },
  });
  const button = el("button", { name: "delete" }, [460, 100, 60, 30]);
  const { body } = withSidebar(p, before, field, button);
  p.load(body);
  // In from the right, the caret is at the text's end: Right goes straight back.
  button.focus();
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement, field);
  p.press("ArrowRight");
  assert.equal(p.document.activeElement, button);
  // In from the left, it is at the start: Left goes straight back.
  before.focus();
  p.press("ArrowRight");
  assert.equal(p.document.activeElement, field);
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement, before);
});

test("a list steps in order, whatever the boxes say", () => {
  const p = page();
  const { el } = p;
  // The second item sits to the right of the first, not below it.
  const items = [
    el("a", { href: "/a", name: "a" }, [100, 100, 50, 20]),
    el("a", { href: "/b", name: "b" }, [400, 100, 50, 20]),
    el("a", { href: "/c", name: "c" }, [100, 130, 50, 20]),
  ];
  const list = el("dd", { "data-nav": "list" }, [100, 100, 400, 60], ...items);
  const { body } = withSidebar(p, list);
  p.load(body);
  items[0].focus();
  p.press("ArrowDown");
  assert.equal(p.document.activeElement.attrs.name, "b");
  p.press("ArrowDown");
  assert.equal(p.document.activeElement.attrs.name, "c");
});

test("a grid moves by its columns", () => {
  const p = page();
  const { el } = p;
  const tiles = [0, 1, 2, 3, 4, 5].map((i) =>
    el("a", { href: `/t${i}`, "data-nav": "grid", name: `t${i}` }, [
      100 + 120 * (i % 3),
      100 + 200 * Math.floor(i / 3),
      100,
      180,
    ]),
  );
  const grid = el(
    "div",
    { class: "grid", "data-nav-main": "" },
    [100, 100, 360, 380],
    ...tiles,
  );
  const { body } = withSidebar(p, grid);
  p.load(body);
  tiles[1].focus();
  p.press("ArrowDown");
  assert.equal(p.document.activeElement.attrs.name, "t4");
  p.press("ArrowRight");
  assert.equal(p.document.activeElement.attrs.name, "t5");
});

test("only Left enters the sidebar, at the page's own link; Right returns to the content", () => {
  const p = page();
  const { el } = p;
  // A header button sits above the tiles, level with the sidebar's first link.
  const header = el("a", { href: "/h", name: "header" }, [80, 0, 60, 30]);
  const tiles = [0, 1].map((i) =>
    el("a", { href: `/t${i}`, "data-nav": "grid", name: `t${i}` }, [
      80 + 120 * i,
      100,
      100,
      180,
    ]),
  );
  const grid = el(
    "div",
    { class: "grid", "data-nav-main": "" },
    [80, 100, 240, 180],
    ...tiles,
  );
  const { body, links } = withSidebar(p, header, grid);
  p.load(body);
  tiles[1].focus();
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement.attrs.name, "t0");
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement, links[1]);
  // Up and Down walk the sidebar and stop at its ends; Left stays.
  p.press("ArrowDown");
  p.press("ArrowDown");
  assert.equal(p.document.activeElement, links[2]);
  p.press("ArrowLeft");
  assert.equal(p.document.activeElement, links[2]);
  // Right goes back to where focus left the content.
  p.press("ArrowRight");
  assert.equal(p.document.activeElement.attrs.name, "t0");
  // Up from the content reaches the header, never the sidebar beside it.
  p.press("ArrowUp");
  assert.equal(p.document.activeElement.attrs.name, "header");
});

test("Up or Down with only the sidebar that way stays put", () => {
  const p = page();
  const { el } = p;
  // Nothing in the content above or below the tile; sidebar links are both.
  const tile = el(
    "a",
    { href: "/t", "data-nav": "grid", name: "t" },
    [80, 120, 100, 40],
  );
  const { body } = withSidebar(p, tile);
  p.load(body);
  tile.focus();
  p.press("ArrowUp");
  p.press("ArrowDown");
  assert.equal(p.document.activeElement.attrs.name, "t");
});

test("a page whose tiles load late hands focus to the first tile when it arrives", () => {
  const p = page();
  const { el } = p;
  const header = el("a", { href: "/search", name: "search" }, [80, 0, 60, 30]);
  const grid = el("div", { class: "grid", "data-nav-main": "" }, [80, 100, 400, 400]);
  const { body } = withSidebar(p, header, grid);
  p.load(body);
  // Nothing to show yet: the header holds focus meanwhile.
  assert.equal(p.document.activeElement.attrs.name, "search");
  grid.append(
    el("a", { href: "/m1", "data-nav": "grid", name: "m1" }, [80, 100, 100, 180]),
  );
  assert.equal(p.document.activeElement.attrs.name, "m1");
});

test("late tiles leave focus where the user already moved it", () => {
  const p = page();
  const { el } = p;
  const header = el("a", { href: "/search", name: "search" }, [80, 0, 60, 30]);
  const other = el("a", { href: "/fav", name: "favorites" }, [160, 0, 60, 30]);
  const grid = el("div", { class: "grid", "data-nav-main": "" }, [80, 100, 400, 400]);
  const { body } = withSidebar(p, header, other, grid);
  p.load(body);
  p.press("ArrowRight");
  assert.equal(p.document.activeElement.attrs.name, "favorites");
  grid.append(
    el("a", { href: "/m1", "data-nav": "grid", name: "m1" }, [80, 100, 100, 180]),
  );
  assert.equal(p.document.activeElement.attrs.name, "favorites");
});

test("Backspace leaves a page that is not a root page", () => {
  const p = page();
  const { el } = p;
  const play = el("a", { href: "/play", name: "play" }, [80, 0, 60, 30]);
  const { body } = withSidebar(p, play);
  p.load(body);
  p.window.location.pathname = "/play/k3f9x2abqe";
  p.press("Backspace");
  assert.equal(p.history.backs, 1);
});
