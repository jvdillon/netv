// Shared Favorites Grid Module for VOD/Series pages
// Requires: window.FAVORITES_CONFIG = { type: 'movies'|'series', favorites, cardClass, tileClass, detailUrl, orderKey }

(function() {
  'use strict';

  const cfg = window.FAVORITES_CONFIG;
  if (!cfg) return;

  function escapeHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function escapeAttr(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  window.favorites = cfg.favorites;

  function getFavorites() {
    return window.favorites[cfg.type] || {};
  }

  function saveFavorites() {
    fetch('/api/user-prefs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({favorites: window.favorites})
    });
  }

  window.toggleFavorite = function(id, name, cover, ext) {
    const favs = window.favorites[cfg.type];
    if (favs[id]) {
      delete favs[id];
    } else {
      favs[id] = cfg.type === 'movies' ? { name, cover, ext } : { name, cover };
    }
    saveFavorites();
    updateFavoriteButtons();
    if (typeof window.renderFavorites === 'function') window.renderFavorites();
  };

  window.updateFavoriteButtons = function() {
    const favs = getFavorites();
    document.querySelectorAll('.fav-btn').forEach(btn => {
      const card = btn.closest('.' + cfg.cardClass);
      const id = card?.dataset[cfg.type === 'movies' ? 'movieId' : 'seriesId'];
      btn.textContent = favs[id] ? '★' : '☆';
      btn.classList.toggle('text-yellow-400', !!favs[id]);
    });
  };

  // Browse view handlers
  if (cfg.isBrowseView) {
    window.updateBrowseUrl = function() {
      const cat = document.getElementById('category-select').value;
      const sort = document.getElementById('sort-select').value;
      const params = new URLSearchParams();
      if (cat) params.set('category', cat);
      params.set('sort', sort);
      window.location.href = cfg.baseUrl + '?' + params;
    };

    const catSel = document.getElementById('category-select');
    const sortSel = document.getElementById('sort-select');
    if (catSel) catSel.addEventListener('change', updateBrowseUrl);
    if (sortSel) sortSel.addEventListener('change', updateBrowseUrl);
    updateFavoriteButtons();
  }

  // Favorites view handlers
  if (!cfg.isBrowseView) {
    async function getOrder() {
      try {
        const resp = await fetch('/api/settings');
        const settings = await resp.json();
        return settings[cfg.orderKey] || [];
      } catch (e) { console.error('Failed to get order:', e); return []; }
    }

    async function saveOrder(order) {
      try {
        const resp = await fetch('/api/settings');
        const settings = await resp.json();
        settings[cfg.orderKey] = order;
        await fetch('/api/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(settings)
        });
      } catch (e) {
        console.error('Failed to save order:', e);
      }
    }

    window.renderFavorites = async function() {
      const favs = getFavorites();
      const grid = document.getElementById('favorites-grid');
      const noFavs = document.getElementById('no-favorites');

      let ids = Object.keys(favs);
      if (ids.length === 0) {
        grid.innerHTML = '';
        noFavs.classList.remove('hidden');
        return;
      }

      const order = await getOrder();
      const orderedIds = order.filter(id => favs[id]);
      const unorderedIds = ids.filter(id => !order.includes(id));
      ids = [...orderedIds, ...unorderedIds];

      noFavs.classList.add('hidden');
      grid.innerHTML = ids.map(id => {
        const f = favs[id];
        const safeId = escapeAttr(id);
        const safeCover = escapeAttr(f.cover);
        const safeName = escapeHtml(f.name);
        return `
          <div class="${cfg.tileClass}" data-id="${safeId}">
            <a href="${cfg.detailUrl}${encodeURIComponent(id)}" class="block bg-gray-800 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-500 focus:ring-2 focus:ring-blue-500 focusable group relative"
               tabindex="0" data-nav="grid">
              <div class="aspect-[2/3] bg-gray-700">
                ${f.cover ? `<img src="${safeCover}" class="w-full h-full object-cover" loading="lazy">` : ''}
              </div>
              <button class="absolute top-2 right-2 w-8 h-8 rounded-full bg-black/50 text-xl text-yellow-400 opacity-0 group-hover:opacity-100 focus:opacity-100 z-10 focusable"
                      tabindex="0"
                      onclick="event.preventDefault(); event.stopPropagation(); toggleFavorite('${safeId}', '', '', '');">★</button>
              <div class="p-2 text-sm line-clamp-2">${safeName}</div>
            </a>
          </div>
        `;
      }).join('');

      initDragDrop();
    };

    function initDragDrop() {
      const grid = document.getElementById('favorites-grid');
      let draggedEl = null;
      let touchStartY = 0;
      let touchStartX = 0;
      let longPressTimer = null;
      let isDragging = false;

      grid.addEventListener('contextmenu', (e) => {
        if (e.target.closest('.' + cfg.tileClass)) e.preventDefault();
      });

      grid.querySelectorAll('.' + cfg.tileClass).forEach(tile => {
        tile.draggable = true;

        tile.addEventListener('dragstart', () => {
          draggedEl = tile;
          tile.classList.add('opacity-50');
        });

        tile.addEventListener('dragend', () => {
          tile.classList.remove('opacity-50');
          draggedEl = null;
          saveCurrentOrder();
        });

        tile.addEventListener('dragover', (e) => {
          e.preventDefault();
          if (draggedEl && draggedEl !== tile) {
            const rect = tile.getBoundingClientRect();
            const midpoint = rect.left + rect.width / 2;
            grid.insertBefore(draggedEl, e.clientX < midpoint ? tile : tile.nextSibling);
          }
        });

        tile.addEventListener('touchstart', (e) => {
          touchStartX = e.touches[0].clientX;
          touchStartY = e.touches[0].clientY;
          longPressTimer = setTimeout(() => {
            isDragging = true;
            draggedEl = tile;
            tile.classList.add('opacity-50', 'ring-2', 'ring-blue-500');
            navigator.vibrate?.(50);
          }, 400);
        }, {passive: true});

        tile.addEventListener('touchmove', (e) => {
          if (longPressTimer && !isDragging) {
            const dx = Math.abs(e.touches[0].clientX - touchStartX);
            const dy = Math.abs(e.touches[0].clientY - touchStartY);
            if (dx > 10 || dy > 10) { clearTimeout(longPressTimer); longPressTimer = null; }
          }
          if (!isDragging || !draggedEl) return;
          e.preventDefault();
          const touch = e.touches[0];
          const target = document.elementFromPoint(touch.clientX, touch.clientY)?.closest('.' + cfg.tileClass);
          if (target && target !== draggedEl) {
            const rect = target.getBoundingClientRect();
            const midpoint = rect.left + rect.width / 2;
            grid.insertBefore(draggedEl, touch.clientX < midpoint ? target : target.nextSibling);
          }
        }, {passive: false});

        tile.addEventListener('touchend', () => {
          clearTimeout(longPressTimer);
          longPressTimer = null;
          if (isDragging && draggedEl) {
            draggedEl.classList.remove('opacity-50', 'ring-2', 'ring-blue-500');
            draggedEl = null;
            isDragging = false;
            saveCurrentOrder();
          }
        });
      });
    }

    async function saveCurrentOrder() {
      const grid = document.getElementById('favorites-grid');
      const order = Array.from(grid.querySelectorAll('.' + cfg.tileClass)).map(tile => tile.dataset.id);
      await saveOrder(order);
    }

    renderFavorites();
  }
})();
