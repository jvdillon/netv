// Keyboard navigation for 10-foot UI: the one definition of what arrows do.
(() => {
  // A region whose items are not all in the page (the guide's virtual rows) or whose
  // arrows mean something else (the player's seek and volume) moves focus itself:
  // its move(current, event) returns true when it took the key, calling
  // preventDefault itself if the browser's default should not run.
  const regions = [];
  window.netvNav = {
    region(element, move) {
      if (element) regions.push({ element, move });
    },
  };

  function regionTook(current, e) {
    return regions.some(
      ({ element, move }) =>
        (current === document.body || element.contains(current)) && move(current, e),
    );
  }

  // ============================================================
  // Focus Management
  // ============================================================

  function getFocusables(container = document) {
    return Array.from(
      container.querySelectorAll(
        'a[href]:not([disabled]), button:not([disabled]), input:not([disabled]), select:not([disabled]), summary, [tabindex="0"], .focusable',
      ),
    ).filter((el) =>
      // Shown only: a collapsed <details> hides all but its summary.
      el.checkVisibility ? el.checkVisibility() : el.offsetParent !== null,
    );
  }

  function getGridInfo(element) {
    const grid = element.closest(".grid");
    if (!grid) return null;

    const items = Array.from(grid.querySelectorAll('[data-nav="grid"]'));
    const index = items.indexOf(element);
    if (index === -1) return null;

    // Detect columns by comparing Y positions
    let cols = 1;
    if (items.length > 1) {
      const firstTop = items[0].getBoundingClientRect().top;
      for (let i = 1; i < items.length; i++) {
        if (items[i].getBoundingClientRect().top > firstTop + 5) {
          cols = i;
          break;
        }
      }
      if (cols === 1) cols = items.length;
    }

    return { items, index, cols };
  }

  // The item of the grid above or below (browse rows) nearest `current`'s column.
  function adjacentGridItem(current, direction) {
    const grids = Array.from(document.querySelectorAll("main .grid")).filter((g) =>
      g.querySelector('[data-nav="grid"]'),
    );
    const index = grids.indexOf(current.closest(".grid"));
    const target = grids[direction === "down" ? index + 1 : index - 1];
    if (index === -1 || !target) return null;
    const x = current.getBoundingClientRect().left;
    let best = null;
    let bestDist = Infinity;
    for (const item of target.querySelectorAll('[data-nav="grid"]')) {
      const dist = Math.abs(item.getBoundingClientRect().left - x);
      if (dist < bestDist) {
        best = item;
        bestDist = dist;
      }
    }
    return best;
  }

  // The focusable whose center lies most nearly straight ahead in `direction`:
  // distance along it, plus twice the sideways offset, so a control in line beats a
  // nearer one off to the side. Card stars are left to "f".
  function nearestInDirection(current, focusables, direction) {
    if (!current || current === document.body) return null;
    const from = current.getBoundingClientRect();
    const cx = from.left + from.width / 2;
    const cy = from.top + from.height / 2;
    const vertical = direction === "up" || direction === "down";
    let best = null;
    let bestScore = Infinity;
    let bestAcross = Infinity;
    for (const el of focusables) {
      if (el === current || el.classList.contains("fav-btn") || current.contains(el))
        continue;
      // Up or down, another row (data-nav="row") is one stop: it is as near as its
      // whole box, however small and far aside its items sit.
      const row = vertical ? el.closest('[data-nav="row"]') : null;
      const r = (row && !row.contains(current) ? row : el).getBoundingClientRect();
      const dx =
        row && vertical
          ? Math.max(r.left - cx, 0, cx - r.right)
          : r.left + r.width / 2 - cx;
      const dy = r.top + r.height / 2 - cy;
      const ahead = { up: -dy, down: dy, left: -dx, right: dx }[direction];
      const aside = vertical ? Math.abs(dx) : Math.abs(dy);
      // Ahead means past the current element's edge, not merely offset by a pixel.
      const edge =
        direction === "up" || direction === "down" ? from.height / 2 : from.width / 2;
      if (ahead <= edge * 0.5) continue;
      const score = ahead + 2 * aside;
      // A row's items tie on its box; the one nearest across is entered.
      const across = Math.abs(
        el.getBoundingClientRect().left + el.offsetWidth / 2 - cx,
      );
      if (score < bestScore || (score === bestScore && across < bestAcross)) {
        best = el;
        bestScore = score;
        bestAcross = across;
      }
    }
    return best;
  }

  const OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };

  // Where an arrow goes from `current`, or null to stay. One definition for every page:
  // - a list (data-nav="list"): Up/Down step through its items in order;
  // - a row (data-nav="row"): Left/Right step through its items in reading order,
  //   across the lines it wraps onto; Up/Down leave it;
  // - a grid of tiles: by row and column, then into the neighboring grid;
  // - anything else: the nearest focusable that way.
  // Only Left enters the sidebar, at the page's own link; nothing wraps.
  function moveTarget(current, direction) {
    const vertical = direction === "up" || direction === "down";
    const forward = direction === "down" || direction === "right";
    const focusables = getFocusables().filter(
      (el) => direction === "left" || !el.closest("nav") || current?.closest("nav"),
    );

    const list = current?.parentElement?.closest('[data-nav="list"]');
    if (list && vertical) {
      const items = getFocusables(list);
      const next = items[items.indexOf(current) + (forward ? 1 : -1)];
      if (next) return next;
    }

    const row = current?.closest('[data-nav="row"]');
    if (row && !vertical) {
      const items = getFocusables(row);
      const next = items[items.indexOf(current) + (forward ? 1 : -1)];
      if (next) return next;
    }

    const gridInfo = getGridInfo(current);
    if (gridInfo && gridInfo.cols > 1) {
      const { items, index, cols } = gridInfo;
      const step = vertical ? cols : 1;
      const next = items[index + (forward ? step : -step)];
      if (next) return next;
      if (vertical) {
        const adjacent = adjacentGridItem(current, direction);
        if (adjacent) return adjacent;
      }
    }

    const candidates =
      row && vertical ? focusables.filter((el) => !row.contains(el)) : focusables;
    const next = nearestInDirection(current, candidates, direction);
    if (next?.closest("nav") && !current?.closest("nav"))
      return sidebarLinks()[sidebarHome()];
    return next;
  }

  // The last move, so its opposite returns where it came from even when position
  // alone would land elsewhere (leaving a long row, a grid's ragged edge).
  let lastMove = null;
  document.addEventListener("focusin", (e) => {
    // Focus moved some other way (a click, the guide, the sidebar): forget it.
    if (e.target !== lastMove?.to) lastMove = null;
  });

  function moveFocus(direction) {
    const current = document.activeElement;
    const back =
      lastMove?.to === current &&
      lastMove.direction === OPPOSITE[direction] &&
      lastMove.from;
    const next =
      back?.isConnected && back.checkVisibility()
        ? back
        : moveTarget(current, direction);
    if (!next) return false;
    lastMove = { from: current, to: next, direction };
    next.focus();
    // A text field takes the caret at the side it was entered from, so the opposite
    // arrow leaves it again at once; Up and Down leave the caret at the end.
    if (typeof next.selectionStart === "number" && next.type !== "number") {
      const at = direction === "right" ? 0 : next.value.length;
      next.setSelectionRange(at, at);
    }
    next.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    return true;
  }

  // ============================================================
  // Sidebar
  // ============================================================

  // A page marks its content (guide rows, tiles, results, episodes) data-nav-main;
  // its headers (categories, Now, Search, sorting) are a move Up from there.

  // The content's first item, or where focus last was in it.
  let lastMainFocus = null;
  document.addEventListener("focusin", (e) => {
    if (e.target.closest?.("[data-nav-main]")) lastMainFocus = e.target;
  });

  function mainContentItem() {
    // Rows a page re-renders replace the element focus left.
    if (lastMainFocus?.isConnected && lastMainFocus.checkVisibility())
      return lastMainFocus;
    for (const region of document.querySelectorAll("[data-nav-main]")) {
      const first = getFocusables(region)[0];
      if (first) return first;
    }
    return null;
  }

  // The sidebar is a column of its own: Up/Down step through it and stop at its
  // ends, Left stays, Right goes into the page's content.
  const sidebarLinks = () => getFocusables(document.querySelector("nav") || undefined);
  const sidebarHome = () =>
    Math.max(
      0,
      sidebarLinks().findIndex((el) => el.classList.contains("active")),
    );

  function moveInSidebar(key) {
    const links = sidebarLinks();
    const i = links.indexOf(document.activeElement);
    if (key === "ArrowUp" || key === "ArrowDown") {
      links[i + (key === "ArrowDown" ? 1 : -1)]?.focus();
    } else if (key === "ArrowRight") {
      const item = mainContentItem();
      if (item) item.focus({ preventScroll: true });
      else focusPageStart();
    }
  }

  // ============================================================
  // Initial Focus
  // ============================================================

  // Priority: [autofocus] or [data-initial-focus] (a search field, Play, the selected
  // guide category), the page's content, its first focusable. The page stays at its
  // top: focus far down would scroll the title away.
  // Set when the page started before its content had loaded (favorites, browse rows
  // fill in later): the first content item to appear then takes focus, unless the
  // user has moved it first.
  let awaitingContent = false;

  function focusPageStart() {
    const main = document.querySelector("main");
    const content =
      document.querySelector("[autofocus], [data-initial-focus]") || mainContentItem();
    awaitingContent = !content;
    (content || (main && getFocusables(main)[0]))?.focus({ preventScroll: true });
  }

  new MutationObserver(() => {
    const item = awaitingContent && mainContentItem();
    if (!item) return;
    awaitingContent = false;
    item.focus({ preventScroll: true });
  }).observe(document.body, { childList: true, subtree: true });
  for (const kind of ["keydown", "pointerdown"])
    document.addEventListener(kind, () => (awaitingContent = false), true);

  function setInitialFocus() {
    // Skip if something is already focused (other than body)
    if (document.activeElement && document.activeElement !== document.body) return;
    focusPageStart();
  }

  // ============================================================
  // Favorites Toggle
  // ============================================================

  function toggleFocusedFavorite() {
    const el = document.activeElement;
    if (!el) return false;

    // A card from _cards.html; a channel's card has no star.
    const card = el.closest(".movie-card, .series-card, .live-card");
    if (card) {
      card.querySelector(".fav-btn")?.click();
      return true;
    }

    // Check for detail page favorite button
    const favBtn = document.getElementById("fav-btn");
    if (favBtn) {
      favBtn.click();
      return true;
    }

    return false;
  }

  // ============================================================
  // Keyboard Handler
  // ============================================================

  document.addEventListener("keydown", (e) => {
    if (
      e.target.closest?.("nav") &&
      !e.altKey &&
      ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)
    ) {
      e.preventDefault();
      moveInSidebar(e.key);
      return;
    }
    if (
      !e.altKey &&
      [
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "PageUp",
        "PageDown",
        "Home",
        "End",
      ].includes(e.key) &&
      regionTook(document.activeElement, e)
    ) {
      return;
    }
    const isInput = e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA";
    const isSelect = e.target.tagName === "SELECT";

    // Arrows and Enter alone reach and operate every control. In a field that takes
    // text, Left/Right move the caret and Up/Down leave; checkboxes and selects
    // take every arrow as movement, and Enter ticks a box or opens a select's list.
    if (isInput || isSelect) {
      if (e.key === "Escape") {
        e.target.blur();
        return;
      }
      const toggles = e.target.type === "checkbox" || e.target.type === "radio";
      if (e.key === "Enter" && (toggles || isSelect)) {
        e.preventDefault();
        if (isSelect) e.target.showPicker?.();
        else e.target.click();
        return;
      }
      // Left/Right leave a text field once the caret is at that end of the text; a
      // number field has no caret to move.
      const t = e.target;
      const caret = typeof t.selectionStart === "number" && t.type !== "number";
      const atStart = !caret || (t.selectionStart === 0 && t.selectionEnd === 0);
      const atEnd = !caret || t.selectionStart === t.value.length;
      const leaves =
        e.key === "ArrowUp" ||
        e.key === "ArrowDown" ||
        (e.key === "ArrowLeft" && (toggles || isSelect || atStart)) ||
        (e.key === "ArrowRight" && (toggles || isSelect || atEnd));
      if (!leaves || e.target.tagName === "TEXTAREA" || e.altKey) return;
      e.preventDefault();
      moveFocus(e.key.replace("Arrow", "").toLowerCase());
      return;
    }

    switch (e.key) {
      case "ArrowUp":
      case "ArrowDown":
      case "ArrowLeft":
      case "ArrowRight": {
        // Alt+arrow is the browser's back and forward.
        if (e.altKey) return;
        e.preventDefault();
        const dir = e.key.replace("Arrow", "").toLowerCase();
        moveFocus(dir);
        break;
      }

      case "Enter": {
        const el = document.activeElement;
        if (el?.href) {
          e.preventDefault();
          if (e.ctrlKey || e.metaKey) {
            window.open(el.href, "_blank");
          } else {
            window.location.href = el.href;
          }
        } else if (el?.click && el.tagName !== "A" && el.tagName !== "BUTTON") {
          e.preventDefault();
          el.click();
        }
        break;
      }

      case "f":
      case "F":
        // Ctrl+F and the like belong to the browser (Find).
        if (e.ctrlKey || e.metaKey || e.altKey) break;
        if (toggleFocusedFavorite()) {
          e.preventDefault();
        }
        break;

      case "Escape":
        // Only handle if focus is on a known focusable element (not during browser find dialog, etc.)
        if (!document.activeElement || document.activeElement === document.body) return;
        e.preventDefault();
        if (document.activeElement?.closest("nav")) {
          // In nav - go to main content
          const mainFocusable = document.querySelector(
            'main [data-nav="grid"], main .focusable, main a[href], main button',
          );
          if (mainFocusable) mainFocusable.focus();
        } else {
          // In content - go to nav
          const navLink = document.querySelector("nav .nav-link");
          if (navLink) navLink.focus();
        }
        break;

      case "Backspace": {
        // Go back unless on root pages or in input
        const rootPages = ["/", "/guide", "/vod", "/series", "/search", "/settings"];
        if (!rootPages.includes(location.pathname)) {
          e.preventDefault();
          history.back();
        }
        break;
      }
    }
  });

  // ============================================================
  // Favorites
  // ============================================================

  // One change per request, so a page loaded before another change cannot undo it.
  window.setFavorite = async (kind, id, entry) => {
    const resp = await fetch("/api/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, id: String(id), entry }),
    });
    if (!resp.ok) throw new Error(`Saving favorite failed: ${resp.status}`);
    return (await resp.json()).favorites;
  };

  // Back/Forward shows the page as first loaded; give pages the current favorites.
  window.addEventListener("pageshow", async (e) => {
    if (typeof window.onFavoritesChanged !== "function") return;
    const nav = performance.getEntriesByType("navigation")[0];
    if (!e.persisted && nav?.type !== "back_forward") return;
    const resp = await fetch("/api/user-prefs");
    if (resp.ok) window.onFavoritesChanged((await resp.json()).favorites);
  });

  // Set initial focus after page load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setInitialFocus);
  } else {
    setTimeout(setInitialFocus, 0);
  }
})();
