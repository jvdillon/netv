// Settings Page Module
(function() {
  'use strict';

  const cfg = window.SETTINGS_CONFIG || {};

  // ============================================================
  // Shared Helpers
  // ============================================================

  function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function showFeedback(el, success) {
    if (!el) return;
    const cls = success ? 'ring-green-500' : 'ring-red-500';
    el.classList.add('ring-2', cls);
    setTimeout(() => el.classList.remove('ring-2', cls), success ? 500 : 1000);
  }

  async function saveWithFeedback(url, options, feedbackEl) {
    try {
      const resp = await fetch(url, options);
      showFeedback(feedbackEl, resp.ok);
      return resp;
    } catch (e) {
      console.error('Save failed:', e);
      showFeedback(feedbackEl, false);
      return null;
    }
  }

  function getFeedbackEl(el) {
    if (!el) return null;
    if (el.type === 'radio' || el.type === 'checkbox') return el.closest('label') || el;
    return el;
  }

  // ============================================================
  // Drag-Drop Helper
  // ============================================================

  function setupDragDrop(containerSelector, chipSelector, onDrop) {
    let draggedChip = null;

    document.querySelectorAll(chipSelector).forEach(chip => {
      chip.addEventListener('dragstart', e => {
        draggedChip = chip;
        e.dataTransfer.effectAllowed = 'move';
        chip.classList.add('opacity-50');
      });
      chip.addEventListener('dragend', () => {
        chip.classList.remove('opacity-50');
        draggedChip = null;
      });
      chip.addEventListener('dragover', e => {
        e.preventDefault();
        if (draggedChip && draggedChip !== chip) {
          chip.classList.add('border-t-2', 'border-blue-500');
        }
      });
      chip.addEventListener('dragleave', () => {
        chip.classList.remove('border-t-2', 'border-blue-500');
      });
      chip.addEventListener('drop', e => {
        e.preventDefault();
        e.stopPropagation();
        chip.classList.remove('border-t-2', 'border-blue-500');
        if (draggedChip && draggedChip !== chip) {
          chip.parentElement.insertBefore(draggedChip, chip);
          onDrop?.(chip.parentElement, draggedChip);
        }
      });
    });

    document.querySelectorAll(containerSelector).forEach(container => {
      container.addEventListener('dragover', e => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        container.classList.add('border-blue-500');
      });
      container.addEventListener('dragleave', e => {
        if (!container.contains(e.relatedTarget)) {
          container.classList.remove('border-blue-500');
        }
      });
      container.addEventListener('drop', e => {
        e.preventDefault();
        container.classList.remove('border-blue-500');
        if (draggedChip && draggedChip.parentElement !== container) {
          container.appendChild(draggedChip);
          onDrop?.(container, draggedChip);
        }
      });
    });
  }

  function setupSearch(inputId, clearBtnId, chipSelector) {
    const input = document.getElementById(inputId);
    const clearBtn = document.getElementById(clearBtnId);
    if (!input) return;

    function apply() {
      const q = input.value.toLowerCase();
      document.querySelectorAll(chipSelector).forEach(el => {
        el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
      });
      clearBtn?.classList.toggle('hidden', !input.value);
    }

    input.addEventListener('input', apply);
    clearBtn?.addEventListener('click', () => { input.value = ''; apply(); });
  }

  // ============================================================
  // Global Functions (used by inline handlers)
  // ============================================================

  window.togglePwdVis = function(btn) {
    const input = btn.parentElement.querySelector('input[type="password"], input[type="text"]');
    if (!input) return;
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    btn.querySelector('.eye-off')?.classList.toggle('hidden', isPassword);
    btn.querySelector('.eye-on')?.classList.toggle('hidden', !isPassword);
  };

  window.toggleSourceFields = function(select) {
    const form = select.closest('form');
    const isXtream = select.value === 'xtream';
    const isEpg = select.value === 'epg';
    form.querySelector('.xtream-fields')?.style.setProperty('display', isXtream ? 'grid' : 'none');
    form.querySelector('.non-epg-only')?.style.setProperty('display', isEpg ? 'none' : 'block');
    form.querySelector('.epg-url-field')?.style.setProperty('display', isEpg ? 'none' : 'block');
  };

  window.showDeleteSelfModal = function() {
    document.getElementById('delete-self-modal')?.classList.remove('hidden');
    const pwInput = document.getElementById('delete-self-password');
    if (pwInput) { pwInput.value = ''; pwInput.focus(); }
    const msg = document.getElementById('delete-self-msg');
    if (msg) msg.textContent = '';
  };

  window.hideDeleteSelfModal = function() {
    document.getElementById('delete-self-modal')?.classList.add('hidden');
  };

  window.submitDeleteSelf = async function(e) {
    e.preventDefault();
    const pw = document.getElementById('delete-self-password')?.value;
    const msgEl = document.getElementById('delete-self-msg');
    if (!pw) return;
    const form = new FormData();
    form.append('password', pw);
    try {
      const resp = await fetch('/settings/users/delete/' + cfg.currentUser, { method: 'POST', body: form });
      if (resp.ok || resp.redirected) {
        window.location.href = '/login';
      } else {
        const data = await resp.json();
        if (msgEl) msgEl.textContent = data.detail || 'Failed';
      }
    } catch (e) {
      console.error('Delete self failed:', e);
      if (msgEl) msgEl.textContent = 'Request failed';
    }
  };

  // ============================================================
  // Add Source Type Select
  // ============================================================

  function setupSourceTypeSelect() {
    const typeSelect = document.getElementById('source-type');
    if (!typeSelect) return;

    typeSelect.addEventListener('change', function() {
      const isXtream = this.value === 'xtream';
      const isM3u = this.value === 'm3u';
      const isEpg = this.value === 'epg';

      document.getElementById('xtream-fields')?.style.setProperty('display', isXtream ? 'grid' : 'none');
      document.getElementById('epg-enabled-field')?.style.setProperty('display', isEpg ? 'none' : 'block');

      const deinterlaceField = document.getElementById('deinterlace-field');
      if (deinterlaceField) {
        deinterlaceField.style.display = isEpg ? 'none' : 'block';
        const cb = deinterlaceField.querySelector('input[name="deinterlace_fallback"]');
        if (cb) cb.checked = isM3u;
      }

      document.getElementById('max-streams-field')?.style.setProperty('display', isEpg ? 'none' : 'block');

      const urlInput = document.querySelector('#add-source-form input[name="url"]');
      if (urlInput) {
        const placeholders = { xtream: 'https://server.com', m3u: 'http://server.com/playlist.m3u', epg: 'http://server.com/epg.xml' };
        urlInput.placeholder = placeholders[this.value] || placeholders.xtream;
      }
    });
  }

  // ============================================================
  // Source Edit Auto-Save
  // ============================================================

  function setupSourceEditForms() {
    document.querySelectorAll('.source-edit-form').forEach(form => {
      const sourceId = form.dataset.sourceId;
      if (!sourceId) return;

      form.querySelectorAll('input, select').forEach(el => {
        if (el.type === 'button' || el.type === 'submit') return;
        el.addEventListener('change', async function() {
          await saveWithFeedback(
            `/settings/edit/${sourceId}`,
            { method: 'POST', body: new FormData(form) },
            getFeedbackEl(this)
          );
        });
      });

      form.addEventListener('submit', e => e.preventDefault());
    });

    // Delete source buttons
    document.querySelectorAll('.delete-source-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const sourceId = btn.dataset.sourceId;
        if (!confirm('Delete this source?')) return;
        btn.disabled = true;
        btn.textContent = 'Deleting...';
        try {
          const resp = await fetch(`/settings/delete/${sourceId}`, { method: 'POST' });
          if (resp.ok) location.reload();
          else throw new Error('Delete failed');
        } catch {
          btn.disabled = false;
          btn.textContent = 'Delete';
        }
      });
    });
  }

  // ============================================================
  // Live TV Category Filter
  // ============================================================

  function setupCategoryFilter() {
    const availableContainer = document.getElementById('available-cats');
    const unavailableContainer = document.getElementById('unavailable-cats');
    if (!availableContainer || !unavailableContainer) return;

    async function save(container) {
      const cats = Array.from(availableContainer.querySelectorAll('.cat-chip')).map(el => el.dataset.id);
      await saveWithFeedback(
        '/settings/guide-filter',
        { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cats}) },
        container
      );
    }

    // Initialize order from config
    const chipById = {};
    unavailableContainer.querySelectorAll('.cat-chip').forEach(el => chipById[el.dataset.id] = el);
    (cfg.selectedCats || []).forEach(catId => {
      if (chipById[catId]) availableContainer.appendChild(chipById[catId]);
    });

    setupDragDrop('#available-cats, #unavailable-cats', '#filters .cat-chip', save);
    setupSearch('cat-search', 'cat-search-clear', '#filters .cat-chip');

    document.getElementById('cat-move-all-right')?.addEventListener('click', async () => {
      availableContainer.querySelectorAll('.cat-chip:not([style*="display: none"])').forEach(c => unavailableContainer.appendChild(c));
      await save(unavailableContainer);
    });

    document.getElementById('cat-move-all-left')?.addEventListener('click', async () => {
      unavailableContainer.querySelectorAll('.cat-chip:not([style*="display: none"])').forEach(c => availableContainer.appendChild(c));
      await save(availableContainer);
    });
  }

  // ============================================================
  // VOD Category Filter
  // ============================================================

  function setupVodCategoryFilter() {
    const availableContainer = document.getElementById('available-vod-cats');
    const unavailableContainer = document.getElementById('unavailable-vod-cats');
    if (!availableContainer || !unavailableContainer) return;

    async function save(container) {
      const cats = Array.from(availableContainer.querySelectorAll('.vod-cat-chip')).map(el => el.dataset.id);
      await saveWithFeedback(
        '/settings/vod-filter',
        { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cats}) },
        container
      );
    }

    // Initialize order from config
    const chipById = {};
    unavailableContainer.querySelectorAll('.vod-cat-chip').forEach(el => chipById[el.dataset.id] = el);
    (cfg.selectedVodCats || []).forEach(catId => {
      if (chipById[catId]) availableContainer.appendChild(chipById[catId]);
    });

    setupDragDrop('#available-vod-cats, #unavailable-vod-cats', '#vod-filters .vod-cat-chip', save);
    setupSearch('vod-cat-search', 'vod-cat-search-clear', '#vod-filters .vod-cat-chip');

    document.getElementById('vod-cat-move-all-right')?.addEventListener('click', async () => {
      availableContainer.querySelectorAll('.vod-cat-chip:not([style*="display: none"])').forEach(c => unavailableContainer.appendChild(c));
      await save(unavailableContainer);
    });

    document.getElementById('vod-cat-move-all-left')?.addEventListener('click', async () => {
      unavailableContainer.querySelectorAll('.vod-cat-chip:not([style*="display: none"])').forEach(c => availableContainer.appendChild(c));
      await save(availableContainer);
    });
  }

  // ============================================================
  // Series Category Filter
  // ============================================================

  function setupSeriesCategoryFilter() {
    const availableContainer = document.getElementById('available-series-cats');
    const unavailableContainer = document.getElementById('unavailable-series-cats');
    if (!availableContainer || !unavailableContainer) return;

    async function save(container) {
      const cats = Array.from(availableContainer.querySelectorAll('.series-cat-chip')).map(el => el.dataset.id);
      await saveWithFeedback(
        '/settings/series-filter',
        { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cats}) },
        container
      );
    }

    // Initialize order from config
    const chipById = {};
    unavailableContainer.querySelectorAll('.series-cat-chip').forEach(el => chipById[el.dataset.id] = el);
    (cfg.selectedSeriesCats || []).forEach(catId => {
      if (chipById[catId]) availableContainer.appendChild(chipById[catId]);
    });

    setupDragDrop('#available-series-cats, #unavailable-series-cats', '#series-filters .series-cat-chip', save);
    setupSearch('series-cat-search', 'series-cat-search-clear', '#series-filters .series-cat-chip');

    document.getElementById('series-cat-move-all-right')?.addEventListener('click', async () => {
      availableContainer.querySelectorAll('.series-cat-chip:not([style*="display: none"])').forEach(c => unavailableContainer.appendChild(c));
      await save(unavailableContainer);
    });

    document.getElementById('series-cat-move-all-left')?.addEventListener('click', async () => {
      unavailableContainer.querySelectorAll('.series-cat-chip:not([style*="display: none"])').forEach(c => availableContainer.appendChild(c));
      await save(availableContainer);
    });
  }

  // ============================================================
  // Chrome CC Link Copy
  // ============================================================

  function setupChromeCcLink() {
    const el = document.getElementById('chrome-cc-link');
    if (!el) return;
    el.addEventListener('click', async () => {
      const text = 'chrome://settings/captions';
      const orig = el.textContent;
      try {
        await navigator.clipboard.writeText(text);
        el.textContent = 'Copied!';
      } catch (e) {
        console.error('Copy failed:', e);
        el.textContent = 'Failed';
      }
      setTimeout(() => el.textContent = orig, 1500);
    });
  }

  // ============================================================
  // Caption Settings
  // ============================================================

  function setupCaptionSettings() {
    const preview = document.getElementById('cc-preview');
    const selects = document.querySelectorAll('.cc-setting');
    const langSelect = document.getElementById('cc-lang-pref');
    const enabledCb = document.getElementById('captions-enabled');

    let ccStyle = cfg.ccStyle || {};

    function hexToRgba(hex, opacity) {
      if (hex === 'transparent') return 'transparent';
      const r = parseInt(hex.slice(1,3), 16);
      const g = parseInt(hex.slice(3,5), 16);
      const b = parseInt(hex.slice(5,7), 16);
      return `rgba(${r},${g},${b},${opacity})`;
    }

    function updatePreview() {
      if (!preview) return;
      preview.style.color = hexToRgba(ccStyle.cc_color || '#ffffff', 1);
      preview.style.textShadow = ccStyle.cc_shadow || '0 0 4px black, 0 0 4px black';
      preview.style.backgroundColor = hexToRgba(ccStyle.cc_bg || '#000000', ccStyle.cc_bg_opacity ?? 0.5);
      preview.style.fontSize = ccStyle.cc_size || '1em';
      preview.style.fontFamily = ccStyle.cc_font || 'inherit';
    }

    selects.forEach(sel => {
      if (ccStyle[sel.dataset.setting]) sel.value = ccStyle[sel.dataset.setting];
      sel.addEventListener('change', async function() {
        ccStyle[this.dataset.setting] = this.value;
        updatePreview();
        await saveWithFeedback(
          '/api/user-prefs',
          { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cc_style: ccStyle}) },
          this
        );
      });
    });

    updatePreview();

    if (langSelect) {
      if (cfg.ccLang) langSelect.value = cfg.ccLang;
      langSelect.addEventListener('change', async function() {
        await saveWithFeedback(
          '/api/user-prefs',
          { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cc_lang: this.value}) },
          this
        );
      });
    }

    if (enabledCb) {
      enabledCb.addEventListener('change', async function() {
        const form = new FormData();
        if (this.checked) form.append('enabled', 'on');
        await saveWithFeedback('/settings/captions', { method: 'POST', body: form }, getFeedbackEl(this));
      });
    }
  }

  // ============================================================
  // Guide Settings
  // ============================================================

  function setupGuideSettings() {
    const virtualScrollCb = document.getElementById('virtual-scroll');
    if (virtualScrollCb) {
      virtualScrollCb.addEventListener('change', async function() {
        await saveWithFeedback(
          '/api/user-prefs',
          { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({virtual_scroll: this.checked}) },
          getFeedbackEl(this)
        );
      });
    }
  }

  // ============================================================
  // Transcode & User-Agent Settings
  // ============================================================

  function setupTranscodeSettings() {
    const container = document.getElementById('transcode-settings');
    if (!container) return;

    // Collect all transcode-related inputs (in container + probe checkboxes + transcode_dir)
    const transcodeInputs = [
      ...container.querySelectorAll('.setting-input'),
      ...document.querySelectorAll('input[name="probe_live"], input[name="probe_movies"], input[name="probe_series"]'),
      document.querySelector('input[name="transcode_dir"]')
    ].filter(Boolean);

    async function save(triggerEl) {
      const form = new FormData();
      // Auto-collect all transcode inputs by type
      transcodeInputs.forEach(el => {
        if (!el.name) return;
        if (el.type === 'checkbox') {
          if (el.checked) form.append(el.name, 'on');
        } else if (el.type === 'radio') {
          if (el.checked) form.append(el.name, el.value);
        } else {
          form.append(el.name, el.value);
        }
      });
      await saveWithFeedback('/settings/transcode', { method: 'POST', body: form }, getFeedbackEl(triggerEl));
    }

    // Auto-enable "Always" transcode when AI Upscale is enabled
    const srInputs = container.querySelectorAll('input[name="sr_model"]');
    const alwaysRadio = container.querySelector('input[name="transcode_mode"][value="always"]');
    srInputs.forEach(sr => {
      sr.addEventListener('change', function() {
        if (this.value !== '' && alwaysRadio && !alwaysRadio.checked) {
          alwaysRadio.checked = true;
          showFeedback(getFeedbackEl(alwaysRadio), true);
        }
      });
    });

    transcodeInputs.forEach(el => {
      el.addEventListener('change', function() { save(this); });
    });

    // Re-detect hardware button
    const refreshBtn = document.getElementById('refresh-encoders-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Detecting...';
        try {
          const resp = await fetch('/settings/refresh-encoders', { method: 'POST' });
          if (resp.ok) {
            const { encoders = {} } = await resp.json();
            // Map encoder detection to radio button enable/disable states
            const radioStates = {
              'nvenc+vaapi': encoders.nvenc && encoders.vaapi,
              'nvenc+software': encoders.nvenc,
              'amf+vaapi': encoders.amf && encoders.vaapi,
              'amf+software': encoders.amf,
              'qsv': encoders.qsv,
              'vaapi': encoders.vaapi,
              'software': true,  // Always available
            };
            Object.entries(radioStates).forEach(([value, enabled]) => {
              const radio = container.querySelector(`input[name="transcode_hw"][value="${value}"]`);
              const label = radio?.closest('label');
              if (radio && label) {
                radio.disabled = !enabled;
                label.classList.toggle('opacity-40', !enabled);
              }
            });
            refreshBtn.textContent = 'Done!';
          } else {
            refreshBtn.textContent = 'Failed';
          }
        } catch (e) {
          console.error('Refresh encoders failed:', e);
          refreshBtn.textContent = 'Failed';
        }
        setTimeout(() => { refreshBtn.textContent = 'Re-detect Hardware'; refreshBtn.disabled = false; }, 1500);
      });
    }
  }

  function setupUserAgentSettings() {
    const container = document.getElementById('user-agent-settings');
    if (!container) return;

    const customContainer = document.getElementById('custom-user-agent-container');
    const customInput = container.querySelector('input[name="user_agent_custom"]');

    async function save(triggerEl) {
      const form = new FormData();
      form.append('preset', container.querySelector('input[name="user_agent_preset"]:checked')?.value || 'default');
      form.append('custom', customInput?.value || '');
      await saveWithFeedback('/settings/user-agent', { method: 'POST', body: form }, getFeedbackEl(triggerEl));
    }

    container.querySelectorAll('input[name="user_agent_preset"]').forEach(radio => {
      radio.addEventListener('change', function() {
        customContainer?.classList.toggle('hidden', this.value !== 'custom');
        save(this);
      });
    });

    customInput?.addEventListener('change', function() { save(this); });
  }

  function setupM3uExportSettings() {
    const container = document.getElementById('m3u-export-settings');
    if (!container) return;

    const enabledCb = container.querySelector('input[name="m3u_export_enabled"]');
    const urlsBox = document.getElementById('m3u-export-urls');

    async function save(triggerEl) {
      const form = new FormData();
      if (enabledCb?.checked) form.append('m3u_export_enabled', 'on');
      form.append('m3u_export_mode', container.querySelector('input[name="m3u_export_mode"]:checked')?.value || 'redirect');
      await saveWithFeedback('/settings/m3u-export', { method: 'POST', body: form }, getFeedbackEl(triggerEl));
      urlsBox?.classList.toggle('hidden', !enabledCb?.checked);
    }

    container.querySelectorAll('.setting-input').forEach(el => {
      el.addEventListener('change', function() { save(this); });
    });
  }

  // ============================================================
  // Data & Probe Cache
  // ============================================================

  function setupDataCache() {
    const clearBtn = document.getElementById('clear-data-cache');
    if (!clearBtn) return;

    clearBtn.addEventListener('click', async () => {
      clearBtn.disabled = true;
      clearBtn.textContent = 'Deleting...';
      const resp = await saveWithFeedback('/settings/data-cache/clear', { method: 'POST' }, clearBtn);
      clearBtn.textContent = resp?.ok ? 'Deleted!' : 'Failed';
      setTimeout(() => { clearBtn.textContent = 'Delete'; clearBtn.disabled = false; }, 2000);
    });
  }

  function setupProbeCache() {
    const listEl = document.getElementById('probe-cache-list');
    const clearAllBtn = document.getElementById('clear-all-probe-cache');
    if (!listEl) return;

    function formatDuration(secs) {
      if (!secs || secs <= 0) return '';
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      return h > 0 ? `${h}h${m}m` : `${m}m`;
    }

    function loadCache() {
      fetch('/settings/probe-cache')
        .then(r => r.json())
        .then(data => {
          const series = data.series || [];
          if (series.length === 0) {
            listEl.innerHTML = '<div class="text-gray-500 text-sm">No cached probes</div>';
            return;
          }
          listEl.innerHTML = series.map(s => {
            const name = escapeHtml(s.name) || `Series ${s.series_id}`;
            const episodes = s.episodes || [];
            const mruEp = s.mru != null ? episodes.find(ep => ep.episode_id === s.mru) : null;
            const mruName = mruEp ? escapeHtml(mruEp.name) || `Episode ${s.mru}` : (s.mru != null ? `Episode ${s.mru}` : null);
            return `
              <details class="bg-gray-700 rounded group">
                <summary class="flex items-center justify-between p-2 cursor-pointer hover:bg-gray-600 rounded text-sm">
                  <div class="flex-1 min-w-0">
                    <span class="font-medium truncate">${name}</span>
                    <span class="text-gray-400 ml-2">${s.episode_count} ep${s.episode_count > 1 ? 's' : ''}</span>
                    <span class="text-gray-500 ml-2">${escapeHtml(s.video_codec || '')}/${escapeHtml(s.audio_codec || '')}</span>
                    ${s.subtitle_count > 0 ? `<span class="text-gray-500 ml-1">+${s.subtitle_count} subs</span>` : ''}
                  </div>
                  <button class="clear-series px-2 py-1 text-xs bg-gray-600 hover:bg-red-600 rounded ml-2" data-series="${s.series_id}">Clear</button>
                </summary>
                <div class="p-2 pt-0 border-t border-gray-600 max-h-48 overflow-y-auto">
                  ${mruName ? `
                    <div class="flex items-center justify-between py-1 text-xs text-blue-400 border-b border-gray-600 mb-1 pb-1">
                      <span class="truncate mr-2">MRU: ${mruName}</span>
                      <button class="clear-mru flex-shrink-0 px-1.5 py-0.5 bg-gray-600 hover:bg-red-600 rounded" data-series="${s.series_id}">×</button>
                    </div>
                  ` : ''}
                  ${episodes.map(ep => `
                    <div class="flex items-center justify-between py-1 text-xs text-gray-400">
                      <span class="truncate mr-2">${escapeHtml(ep.name) || 'Episode ' + ep.episode_id}${ep.duration ? ` (${formatDuration(ep.duration)})` : ''}${ep.subtitle_count ? ` +${ep.subtitle_count} subs` : ''}</span>
                      <button class="clear-episode flex-shrink-0 px-1.5 py-0.5 bg-gray-600 hover:bg-red-600 rounded" data-series="${s.series_id}" data-episode="${ep.episode_id}">×</button>
                    </div>
                  `).join('')}
                </div>
              </details>
            `;
          }).join('');
        })
        .catch(() => {
          listEl.innerHTML = '<div class="text-red-400 text-sm">Failed to load</div>';
        });
    }

    clearAllBtn?.addEventListener('click', () => {
      fetch('/settings/probe-cache/clear', { method: 'POST' }).then(() => loadCache());
    });

    // Event delegation for dynamically created buttons
    listEl.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      e.stopPropagation();

      if (btn.classList.contains('clear-series')) {
        fetch(`/settings/probe-cache/clear/${btn.dataset.series}`, { method: 'POST' }).then(() => loadCache());
      } else if (btn.classList.contains('clear-mru')) {
        fetch(`/settings/probe-cache/clear-mru/${btn.dataset.series}`, { method: 'POST' }).then(() => loadCache());
      } else if (btn.classList.contains('clear-episode')) {
        fetch(`/settings/probe-cache/clear/${btn.dataset.series}?episode_id=${btn.dataset.episode}`, { method: 'POST' }).then(() => loadCache());
      }
    });

    loadCache();
  }

  // ============================================================
  // Source Refresh Buttons
  // ============================================================

  function setupRefreshButtons() {
    const activeRefreshes = new Set();
    let pollInterval = null;

    function updateButtonStates(statuses) {
      const globalStatus = statuses._global || {};
      document.querySelectorAll('[data-source-id]').forEach(container => {
        const sourceId = container.dataset.sourceId;
        const sourceStatuses = statuses[sourceId] || {};
        container.querySelectorAll('.refresh-btn').forEach(btn => {
          const refreshType = btn.dataset.refresh;
          const isActive = !!sourceStatuses[refreshType] || !!globalStatus[refreshType];
          btn.classList.toggle('active', isActive);
          if (isActive) activeRefreshes.add(`${sourceId}_${refreshType}`);
          else activeRefreshes.delete(`${sourceId}_${refreshType}`);
        });
      });
      if (activeRefreshes.size === 0 && pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }

    function pollStatus() {
      fetch('/settings/refresh-status').then(r => r.json()).then(updateButtonStates).catch(() => {});
    }

    function startPolling() {
      if (!pollInterval) {
        pollInterval = setInterval(pollStatus, 1000);
        pollStatus();
      }
    }

    document.querySelectorAll('[data-source-id] .refresh-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const container = btn.closest('[data-source-id]');
        const sourceId = container.dataset.sourceId;
        btn.classList.add('active');
        activeRefreshes.add(`${sourceId}_${btn.dataset.refresh}`);
        fetch(`/settings/refresh/${sourceId}/${btn.dataset.refresh}`, { method: 'POST' })
          .then(() => startPolling())
          .catch(() => btn.classList.remove('active'));
      });
    });

    fetch('/settings/refresh-status').then(r => r.json()).then(statuses => {
      if (Object.keys(statuses).length > 0) {
        updateButtonStates(statuses);
        startPolling();
      }
    }).catch(() => {});
  }

  // ============================================================
  // User Management
  // ============================================================

  function setupUserForms() {
    // Add User form
    const addUserForm = document.getElementById('add-user-form');
    if (addUserForm) {
      setupDragDrop(
        '#add-user-available-groups, #add-user-unavailable-groups',
        '.add-user-group-chip',
        null
      );

      setupSearch('add-user-group-search', 'add-user-group-search-clear', '.add-user-group-chip');

      document.getElementById('add-user-block-all')?.addEventListener('click', () => {
        const avail = document.getElementById('add-user-available-groups');
        const unavail = document.getElementById('add-user-unavailable-groups');
        avail?.querySelectorAll('.add-user-group-chip:not([style*="display: none"])').forEach(c => unavail?.appendChild(c));
      });

      document.getElementById('add-user-allow-all')?.addEventListener('click', () => {
        const avail = document.getElementById('add-user-available-groups');
        const unavail = document.getElementById('add-user-unavailable-groups');
        unavail?.querySelectorAll('.add-user-group-chip:not([style*="display: none"])').forEach(c => avail?.appendChild(c));
      });

      addUserForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const form = new FormData(this);

        const maxStreamsPerSource = {};
        document.querySelectorAll('.add-user-source-max-streams').forEach(inp => {
          const val = parseInt(inp.value) || 0;
          if (val > 0) maxStreamsPerSource[inp.dataset.sourceId] = val;
        });
        form.append('max_streams_per_source', JSON.stringify(maxStreamsPerSource));

        const unavailableGroups = Array.from(
          document.querySelectorAll('#add-user-unavailable-groups .add-user-group-chip')
        ).map(c => c.dataset.groupId);
        form.append('unavailable_groups', JSON.stringify(unavailableGroups));

        const msgEl = document.getElementById('add-user-msg');
        try {
          const resp = await fetch('/settings/users/add', { method: 'POST', body: form });
          if (resp.ok) {
            if (msgEl) { msgEl.textContent = 'Added'; msgEl.className = 'text-sm text-green-400'; }
            this.reset();
            setTimeout(() => location.reload(), 500);
          } else {
            const data = await resp.json();
            if (msgEl) { msgEl.textContent = data.detail || 'Failed'; msgEl.className = 'text-sm text-red-400'; }
          }
        } catch (e) {
          console.error('Add user failed:', e);
          if (msgEl) { msgEl.textContent = 'Request failed'; msgEl.className = 'text-sm text-red-400'; }
        }
        msgEl?.classList.remove('hidden');
        setTimeout(() => { if (msgEl) msgEl.className = 'text-sm hidden'; }, 3000);
      });
    }

    // Password inputs
    document.querySelectorAll('.password-input').forEach(input => {
      input.addEventListener('change', async function() {
        const username = this.closest('[data-username]')?.dataset.username;
        if (!username || this.value.length < 8) {
          showFeedback(this, false);
          return;
        }
        const form = new FormData();
        form.append('new_password', this.value);
        const resp = await saveWithFeedback(`/settings/users/password/${username}`, { method: 'POST', body: form }, this);
        if (resp?.ok) this.value = '';
      });
    });

    // Admin toggles
    document.querySelectorAll('.admin-toggle').forEach(checkbox => {
      checkbox.addEventListener('change', async function() {
        const username = this.closest('[data-username]')?.dataset.username;
        if (!username) return;
        const form = new FormData();
        if (this.checked) form.append('admin', 'on');
        try {
          const resp = await fetch(`/settings/users/admin/${username}`, { method: 'POST', body: form });
          if (resp.ok) location.reload();
          else this.checked = !this.checked;
        } catch (e) {
          console.error('Admin toggle failed:', e);
          this.checked = !this.checked;
        }
      });
    });

    // Max streams per source
    document.querySelectorAll('.user-source-max-streams').forEach(input => {
      input.addEventListener('change', async function() {
        const container = this.closest('.user-max-streams-container');
        const username = container?.dataset.username;
        if (!username) return;

        const maxStreamsPerSource = {};
        container.querySelectorAll('.user-source-max-streams').forEach(inp => {
          const val = parseInt(inp.value) || 0;
          if (val > 0) maxStreamsPerSource[inp.dataset.sourceId] = val;
        });

        const form = new FormData();
        form.append('max_streams_per_source', JSON.stringify(maxStreamsPerSource));
        await saveWithFeedback(`/settings/users/limits/${username}`, { method: 'POST', body: form }, this);
      });
    });

    // Group restrictions
    setupUserGroupDragDrop();
  }

  function setupUserGroupDragDrop() {
    async function saveGroups(username, feedbackContainer) {
      const unavailableContainer = document.querySelector(`.user-unavailable-groups[data-username="${username}"]`);
      const unavailableGroups = Array.from(unavailableContainer?.querySelectorAll('.group-chip') || [])
        .map(c => c.dataset.groupId);

      const form = new FormData();
      form.append('unavailable_groups', JSON.stringify(unavailableGroups));
      await saveWithFeedback(`/settings/users/limits/${username}`, { method: 'POST', body: form }, feedbackContainer);
    }

    setupDragDrop('.user-available-groups, .user-unavailable-groups', '.group-chip', (container) => {
      const username = container.dataset.username;
      if (username) saveGroups(username, container);
    });

    // Search per user
    document.querySelectorAll('.user-group-search').forEach(input => {
      const username = input.dataset.username;
      const clearBtn = document.querySelector(`.user-group-search-clear[data-username="${username}"]`);

      function apply() {
        const q = input.value.toLowerCase();
        [`.user-available-groups[data-username="${username}"]`, `.user-unavailable-groups[data-username="${username}"]`].forEach(sel => {
          document.querySelectorAll(`${sel} .group-chip`).forEach(chip => {
            chip.style.display = chip.textContent.toLowerCase().includes(q) ? '' : 'none';
          });
        });
        clearBtn?.classList.toggle('hidden', !input.value);
      }

      input.addEventListener('input', apply);
      clearBtn?.addEventListener('click', () => { input.value = ''; apply(); });
    });

    // Move all buttons
    document.querySelectorAll('.group-move-all-unavailable').forEach(btn => {
      btn.addEventListener('click', async () => {
        const username = btn.dataset.username;
        if (!username) return;
        const avail = document.querySelector(`.user-available-groups[data-username="${username}"]`);
        const unavail = document.querySelector(`.user-unavailable-groups[data-username="${username}"]`);
        avail?.querySelectorAll('.group-chip:not([style*="display: none"])').forEach(c => unavail?.appendChild(c));
        await saveGroups(username, unavail);
      });
    });

    document.querySelectorAll('.group-move-all-available').forEach(btn => {
      btn.addEventListener('click', async () => {
        const username = btn.dataset.username;
        if (!username) return;
        const avail = document.querySelector(`.user-available-groups[data-username="${username}"]`);
        const unavail = document.querySelector(`.user-unavailable-groups[data-username="${username}"]`);
        unavail?.querySelectorAll('.group-chip:not([style*="display: none"])').forEach(c => avail?.appendChild(c));
        await saveGroups(username, avail);
      });
    });
  }

  // ============================================================
  // Init
  // ============================================================

  function init() {
    setupSourceTypeSelect();
    setupSourceEditForms();
    setupCategoryFilter();
    setupVodCategoryFilter();
    setupSeriesCategoryFilter();
    setupChromeCcLink();
    setupCaptionSettings();
    setupGuideSettings();
    setupTranscodeSettings();
    setupUserAgentSettings();
    setupM3uExportSettings();
    setupDataCache();
    setupProbeCache();
    setupRefreshButtons();
    setupUserForms();
  }

  init();
})();
