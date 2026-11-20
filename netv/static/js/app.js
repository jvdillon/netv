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
        'a[href]:not([disabled]), button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex="0"], .focusable',
      ),
    ).filter((el) => el.offsetParent !== null); // visible only
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

  function moveFocus(direction) {
    const current = document.activeElement;
    const focusables = getFocusables();
    const currentIndex = focusables.indexOf(current);

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
      // At grid edge: step into the neighboring grid (browse rows), never wrap.
      if (direction === "up" || direction === "down") {
        return focusAdjacentGrid(current, direction);
      }
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

    // Priority: [autofocus], first grid item, first focusable in main
    const autofocus = document.querySelector("[autofocus]");
    if (autofocus) {
      autofocus.focus();
      return;
    }

    const mainContent = document.querySelector("main");
    if (!mainContent) return;

    const gridItem = mainContent.querySelector('[data-nav="grid"]');
    if (gridItem) {
      gridItem.focus();
      return;
    }

    const firstFocusable = getFocusables(mainContent)[0];
    if (firstFocusable) firstFocusable.focus();
  }

  // ============================================================
  // Favorites Toggle
  // ============================================================

  function toggleFocusedFavorite() {
    const el = document.activeElement;
    if (!el) return false;

    // Check for movie card
    const movieCard = el.closest(".movie-card");
    if (movieCard) {
      const btn = movieCard.querySelector(".fav-btn, .fav-btn-movie");
      if (btn) {
        btn.click();
        return true;
      }
    }

    // Check for series card
    const seriesCard = el.closest(".series-card");
    if (seriesCard) {
      const btn = seriesCard.querySelector(".fav-btn, .fav-btn-series");
      if (btn) {
        btn.click();
        return true;
      }
    }

    // Check for favorites tile (in favorites view)
    const tile = el.closest(".vod-tile, .series-tile");
    if (tile) {
      const btn = tile.querySelector("button");
      if (btn) {
        btn.click();
        return true;
      }
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

    // Input field handling
    if (isInput) {
      if (e.key === "Escape") {
        e.target.blur();
        return;
      }
      // Allow down arrow to escape search input
      if (e.key === "ArrowDown" && e.target.type === "text") {
        const mainContent = document.querySelector("main");
        const firstResult = mainContent?.querySelector('[data-nav="grid"]');
        if (firstResult) {
          e.preventDefault();
          firstResult.focus();
          return;
        }
      }
      // Let other keys work normally in inputs
      return;
    }

    // Select handling - let arrows work for options
    if (isSelect && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
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
