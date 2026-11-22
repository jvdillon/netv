// Favorite stars on the cards that _cards.html draws, on every page that shows them.
// A page's own script sets window.favorites, and may replace window.onFavoritesChanged.

(() => {
  // A card's kind, id, and the favorite entry it would save.
  function cardFavorite(card) {
    const d = card.dataset;
    if (card.classList.contains("movie-card")) {
      return [
        "movies",
        d.movieId,
        { name: d.movieName, cover: d.movieCover, ext: d.movieExt },
      ];
    }
    return ["series", d.seriesId, { name: d.seriesName, cover: d.seriesCover }];
  }

  window.updateFavoriteButtons = () => {
    document.querySelectorAll(".movie-card, .series-card").forEach((card) => {
      const [kind, id] = cardFavorite(card);
      const on = !!window.favorites?.[kind]?.[id];
      const btn = card.querySelector(".fav-btn");
      btn.textContent = on ? "★" : "☆";
      btn.classList.toggle("text-yellow-400", on);
    });
  };

  window.toggleFavorite = async (button) => {
    const [kind, id, entry] = cardFavorite(button.closest(".movie-card, .series-card"));
    const saved = window.favorites?.[kind]?.[id];
    window.onFavoritesChanged(await setFavorite(kind, id, saved ? null : entry));
  };

  // Drag a card, or long-press one on touch, to move it among its grid's cards. Handlers
  // sit on the document, so cards loaded later move too; the grid then gets
  // "cards-reordered", which a grid that keeps its order (Favorites) listens for.
  const CARD = ".movie-card, .series-card";
  let dragged = null;

  function moveBefore(target, x) {
    if (
      !dragged ||
      !target ||
      target === dragged ||
      target.parentNode !== dragged.parentNode
    ) {
      return;
    }
    const rect = target.getBoundingClientRect();
    target.parentNode.insertBefore(
      dragged,
      x < rect.left + rect.width / 2 ? target : target.nextSibling,
    );
  }

  function drop() {
    if (!dragged) return;
    dragged.classList.remove("opacity-50", "ring-2", "ring-blue-500");
    dragged.parentNode.dispatchEvent(new CustomEvent("cards-reordered"));
    dragged = null;
  }

  document.addEventListener("dragstart", (e) => {
    dragged = e.target.closest?.(CARD) ?? null;
    dragged?.classList.add("opacity-50");
  });
  document.addEventListener("dragover", (e) => {
    const target = e.target.closest?.(CARD);
    if (dragged && target) {
      e.preventDefault();
      moveBefore(target, e.clientX);
    }
  });
  document.addEventListener("dragend", drop);

  let press = null;
  document.addEventListener(
    "touchstart",
    (e) => {
      const card = e.target.closest?.(CARD);
      if (!card) return;
      const { clientX: x, clientY: y } = e.touches[0];
      press = {
        x,
        y,
        timer: setTimeout(() => {
          dragged = card;
          card.classList.add("opacity-50", "ring-2", "ring-blue-500");
          navigator.vibrate?.(50);
        }, 400),
      };
    },
    { passive: true },
  );
  document.addEventListener(
    "touchmove",
    (e) => {
      const { clientX: x, clientY: y } = e.touches[0];
      // Moving before the long press completes is a scroll, not a drag.
      if (press && !dragged && Math.hypot(x - press.x, y - press.y) > 10) {
        clearTimeout(press.timer);
        press = null;
      }
      if (!dragged) return;
      e.preventDefault();
      moveBefore(document.elementFromPoint(x, y)?.closest(CARD), x);
    },
    { passive: false },
  );
  document.addEventListener("touchend", () => {
    if (press) clearTimeout(press.timer);
    press = null;
    drop();
  });
  // A long press on a card drags it; it should not open the context menu.
  document.addEventListener("contextmenu", (e) => {
    if (e.target.closest?.(CARD) && e.pointerType === "touch") e.preventDefault();
  });

  window.onFavoritesChanged = (favorites) => {
    window.favorites = favorites;
    window.updateFavoriteButtons();
  };

  // Page scripts run first, so window.favorites is set by now.
  document.addEventListener("DOMContentLoaded", () => window.updateFavoriteButtons());
})();
