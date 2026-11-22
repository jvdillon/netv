// Browse rows and the Favorites grid of the Movies and Series pages.
// Requires: window.FAVORITES_CONFIG = { type: 'movies'|'series', favorites, cardClass,
//   baseUrl, favoritesUrl, orderKey, isBrowseView }. Cards and their stars: cards.js.

(() => {
  const cfg = window.FAVORITES_CONFIG;
  if (!cfg) return;

  window.favorites = cfg.favorites;
  const idKey = cfg.type === "movies" ? "movieId" : "seriesId";

  window.onFavoritesChanged = (favorites) => {
    window.favorites = favorites;
    updateFavoriteButtons();
    if (typeof window.renderFavorites === "function") window.renderFavorites();
  };

  // Browse view handlers
  if (cfg.isBrowseView) {
    window.updateBrowseUrl = () => {
      const cat = document.getElementById("category-select").value;
      const sort = document.getElementById("sort-select").value;
      const params = new URLSearchParams();
      if (cat) params.set("category", cat);
      params.set("sort", sort);
      window.location.href = `${cfg.baseUrl}?${params}`;
    };

    const catSel = document.getElementById("category-select");
    const sortSel = document.getElementById("sort-select");
    if (catSel) catSel.addEventListener("change", updateBrowseUrl);
    if (sortSel) sortSel.addEventListener("change", updateBrowseUrl);

    const browse = document.getElementById("browse");

    // Fetches cards from `url` into `container` ahead of `before`; returns how many arrived.
    async function loadCards(url, container, before) {
      try {
        const resp = await fetch(url);
        if (!resp.ok) return 0;
        const tpl = document.createElement("template");
        tpl.innerHTML = await resp.text();
        const count = tpl.content.querySelectorAll(`.${cfg.cardClass}`).length;
        container.insertBefore(tpl.content, before || null);
        updateFavoriteButtons();
        return count;
      } catch (e) {
        console.error("Failed to load cards:", e);
        return 0;
      }
    }

    const rowObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const row = entry.target;
          rowObserver.unobserve(row);
          const grid = row.querySelector(".grid");
          loadCards(row.dataset.src, grid, grid.querySelector(".see-all"));
        }
      },
      { root: browse, rootMargin: "600px 0px" },
    );
    document.querySelectorAll(".browse-row[data-src]").forEach((row) => {
      rowObserver.observe(row);
    });

    const pageGrid = document.getElementById("browse-grid");
    const sentinel = document.getElementById("browse-more");
    if (pageGrid && sentinel) {
      let offset = Number(pageGrid.dataset.offset);
      const total = Number(pageGrid.dataset.total);
      let busy = false;
      const pageObserver = new IntersectionObserver(
        async (entries) => {
          if (busy || !entries.some((e) => e.isIntersecting)) return;
          busy = true;
          const count = await loadCards(
            `${pageGrid.dataset.src}&offset=${offset}`,
            pageGrid,
          );
          offset += count;
          busy = false;
          pageObserver.unobserve(sentinel);
          // Re-observing reports the sentinel again, so a still-visible one loads the next page.
          if (count && offset < total) pageObserver.observe(sentinel);
        },
        { root: browse, rootMargin: "800px 0px" },
      );
      if (offset < total) pageObserver.observe(sentinel);
    }
  }

  // Favorites view handlers
  if (!cfg.isBrowseView) {
    async function saveOrder(order) {
      try {
        const resp = await fetch("/api/settings");
        const settings = await resp.json();
        settings[cfg.orderKey] = order;
        await fetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings),
        });
      } catch (e) {
        console.error("Failed to save order:", e);
      }
    }

    // The server draws the favorites, in their saved order, with the one card template.
    window.renderFavorites = async () => {
      const grid = document.getElementById("favorites-grid");
      const resp = await fetch(cfg.favoritesUrl);
      if (!resp.ok) return;
      grid.innerHTML = await resp.text();
      const empty = !grid.querySelector(`.${cfg.cardClass}`);
      document.getElementById("no-favorites").classList.toggle("hidden", !empty);
      updateFavoriteButtons();
    };

    // Any card grid can be reordered (cards.js); only this one keeps its order.
    const grid = document.getElementById("favorites-grid");
    grid.addEventListener("cards-reordered", () => {
      saveOrder(
        Array.from(
          grid.querySelectorAll(`.${cfg.cardClass}`),
          (card) => card.dataset[idKey],
        ),
      );
    });

    renderFavorites();
  }
})();
