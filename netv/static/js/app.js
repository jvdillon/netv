// Keyboard navigation for 10-foot UI
(() => {
  // Pages with custom arrow key handling
  const customNavPages = ["/play/", "/guide"];
  const hasCustomNav = customNavPages.some((p) => location.pathname.startsWith(p));

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

  function focusAdjacentGrid(current, direction) {
    const grids = Array.from(document.querySelectorAll("main .grid")).filter((g) =>
      g.querySelector('[data-nav="grid"]'),
    );
    const index = grids.indexOf(current.closest(".grid"));
    const target = grids[direction === "down" ? index + 1 : index - 1];
    if (index === -1 || !target) return false;
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
    best.focus();
    best.scrollIntoView({
      block: "nearest",
      inline: "nearest",
      behavior: "smooth",
    });
    return true;
  }

  // The focusable whose center lies most nearly straight ahead in `direction`:
  // distance along it, plus twice the sideways offset, so a control in line beats a
  // nearer one off to the side. Card stars are left to "f".
  function nearestInDirection(current, focusables, direction) {
    if (!current || current === document.body) return null;
    const from = current.getBoundingClientRect();
    const cx = from.left + from.width / 2;
    const cy = from.top + from.height / 2;
    let best = null;
    let bestScore = Infinity;
    for (const el of focusables) {
      if (el === current || el.classList.contains("fav-btn") || current.contains(el))
        continue;
      const r = el.getBoundingClientRect();
      const dx = r.left + r.width / 2 - cx;
      const dy = r.top + r.height / 2 - cy;
      const ahead = { up: -dy, down: dy, left: -dx, right: dx }[direction];
      const aside =
        direction === "up" || direction === "down" ? Math.abs(dx) : Math.abs(dy);
      // Ahead means past the current element's edge, not merely offset by a pixel.
      const edge =
        direction === "up" || direction === "down" ? from.height / 2 : from.width / 2;
      if (ahead <= edge * 0.5) continue;
      const score = ahead + 2 * aside;
      if (score < bestScore) {
        best = el;
        bestScore = score;
      }
    }
    return best;
  }

  function moveFocus(direction) {
    const current = document.activeElement;
    const focusables = getFocusables();
    const currentIndex = focusables.indexOf(current);

    // In a list (data-nav="list"), Up/Down step through its items in order; its ends,
    // and Left/Right, move on by position. Lists scroll in their own boxes, where
    // position alone can land in a neighboring list.
    const list = current?.parentElement?.closest('[data-nav="list"]');
    if (list && (direction === "up" || direction === "down")) {
      const items = getFocusables(list);
      const next = items[items.indexOf(current) + (direction === "down" ? 1 : -1)];
      if (next) {
        next.focus();
        next.scrollIntoView({ block: "nearest", behavior: "smooth" });
        return true;
      }
    }

    // Try grid navigation first
    const gridInfo = getGridInfo(current);
    if (gridInfo && gridInfo.cols > 1) {
      const { items, index, cols } = gridInfo;
      let nextIndex = -1;

      switch (direction) {
        case "up":
          nextIndex = index - cols;
          break;
        case "down":
          nextIndex = index + cols;
          break;
        case "left":
          nextIndex = index - 1;
          break;
        case "right":
          nextIndex = index + 1;
          break;
      }

      if (nextIndex >= 0 && nextIndex < items.length) {
        items[nextIndex].focus();
        items[nextIndex].scrollIntoView({
          block: "nearest",
          behavior: "smooth",
        });
        return true;
      }
      // At grid edge: step into the neighboring grid (browse rows), never wrap; with
      // none there, move to whatever lies that way (the search form, the sidebar).
      if (
        (direction === "up" || direction === "down") &&
        focusAdjacentGrid(current, direction)
      ) {
        return true;
      }
    }

    const nearest = nearestInDirection(current, focusables, direction);
    if (nearest) {
      nearest.focus();
      nearest.scrollIntoView({
        block: "nearest",
        inline: "nearest",
        behavior: "smooth",
      });
      return true;
    }

    // Linear navigation fallback
    let nextElement = null;
    if (direction === "up" || direction === "left") {
      nextElement = focusables[currentIndex - 1];
    } else {
      nextElement = focusables[currentIndex + 1];
    }

    if (nextElement) {
      nextElement.focus();
      nextElement.scrollIntoView({ block: "nearest", behavior: "smooth" });
      return true;
    }
    return false;
  }

  // ============================================================
  // Initial Focus
  // ============================================================

  function setInitialFocus() {
    // Skip if something is already focused (other than body)
    if (document.activeElement && document.activeElement !== document.body) return;

    // Priority: [autofocus], [data-initial-focus], first grid item, first focusable in
    // main. The page stays at its top: focus far down would scroll the title away.
    const autofocus = document.querySelector("[autofocus], [data-initial-focus]");
    if (autofocus) {
      autofocus.focus({ preventScroll: true });
      return;
    }

    const mainContent = document.querySelector("main");
    if (!mainContent) return;

    const gridItem = mainContent.querySelector('[data-nav="grid"]');
    if (gridItem) {
      gridItem.focus({ preventScroll: true });
      return;
    }

    const firstFocusable = getFocusables(mainContent)[0];
    if (firstFocusable) firstFocusable.focus({ preventScroll: true });
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
      if (!leaves || e.target.tagName === "TEXTAREA" || hasCustomNav || e.altKey)
        return;
      e.preventDefault();
      moveFocus(e.key.replace("Arrow", "").toLowerCase());
      return;
    }

    switch (e.key) {
      case "ArrowUp":
      case "ArrowDown":
      case "ArrowLeft":
      case "ArrowRight": {
        // Skip if page has custom navigation or Alt pressed (browser nav)
        if (hasCustomNav || e.altKey) return;
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
