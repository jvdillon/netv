// Settings Page Module
// Requires: window.SETTINGS_CONFIG

(function() {
  'use strict';

  const cfg = window.SETTINGS_CONFIG;

  // ============================================================
  // Password Visibility Toggle
  // ============================================================

  window.togglePwdVis = function(btn) {
    const input = btn.parentElement.querySelector('input[type="password"], input[type="text"]');
    const eyeOff = btn.querySelector('.eye-off');
    const eyeOn = btn.querySelector('.eye-on');
    if (!input) return;
    if (input.type === 'password') {
      input.type = 'text';
      if (eyeOff) eyeOff.classList.add('hidden');
      if (eyeOn) eyeOn.classList.remove('hidden');
    } else {
      input.type = 'password';
      if (eyeOff) eyeOff.classList.remove('hidden');
      if (eyeOn) eyeOn.classList.add('hidden');
    }
  };

  // ============================================================
  // Source Type Toggle
  // ============================================================

  window.toggleSourceFields = function(select) {
    const form = select.closest('form');
    const xtreamFields = form.querySelector('.xtream-fields');
    const epgToggle = form.querySelector('.non-epg-only');
    const epgUrlField = form.querySelector('.epg-url-field');
    const isXtream = select.value === 'xtream';
    const isEpg = select.value === 'epg';
    if (xtreamFields) xtreamFields.style.display = isXtream ? 'grid' : 'none';
    if (epgToggle) epgToggle.style.display = isEpg ? 'none' : 'block';
    if (epgUrlField) epgUrlField.style.display = isEpg ? 'none' : 'block';
  };

  // ============================================================
  // Delete Self Modal
  // ============================================================

  window.showDeleteSelfModal = function() {
    document.getElementById('delete-self-modal').classList.remove('hidden');
    document.getElementById('delete-self-password').value = '';
    document.getElementById('delete-self-msg').textContent = '';
    document.getElementById('delete-self-password').focus();
  };

  window.hideDeleteSelfModal = function() {
    document.getElementById('delete-self-modal').classList.add('hidden');
  };

  window.submitDeleteSelf = async function(e) {
    e.preventDefault();
    const pw = document.getElementById('delete-self-password').value;
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
        msgEl.textContent = data.detail || 'Failed';
      }
    } catch (err) {
      msgEl.textContent = 'Request failed';
    }
  };

  // ============================================================
  // Add Source Type Change Handler
  // ============================================================

  function setupSourceTypeSelect() {
    const typeSelect = document.getElementById('source-type');
    if (!typeSelect) return;
    typeSelect.addEventListener('change', function() {
      const isXtream = this.value === 'xtream';
      const isM3u = this.value === 'm3u';
      const isEpg = this.value === 'epg';
      document.getElementById('xtream-fields').style.display = isXtream ? 'grid' : 'none';
      document.getElementById('epg-enabled-field').style.display = isEpg ? 'none' : 'block';
      // Deinterlace: hide for EPG, default checked for M3U (OTA), unchecked for Xtream (IPTV)
      const deinterlaceField = document.getElementById('deinterlace-field');
      if (deinterlaceField) {
        deinterlaceField.style.display = isEpg ? 'none' : 'block';
        const checkbox = deinterlaceField.querySelector('input[name="deinterlace_fallback"]');
        if (checkbox) checkbox.checked = isM3u;
      }
      const urlInput = document.querySelector('#add-source-form input[name="url"]');
      const placeholders = { xtream: 'https://server.com', m3u: 'http://server.com/playlist.m3u', epg: 'http://server.com/epg.xml' };
      urlInput.placeholder = placeholders[this.value] || placeholders.xtream;
    });
  }

  // ============================================================
  // Live TV Category Filter
  // ============================================================

  function setupCategoryFilter() {
    const availableContainer = document.getElementById('available-cats');
    const selectedContainer = document.getElementById('selected-cats');
    const dropHint = document.getElementById('drop-hint');
    const searchInput = document.getElementById('cat-search');
    if (!availableContainer || !selectedContainer) return;

    let orderedCats = cfg.selectedCats || [];

    function updateUI() {
      const selectedSet = new Set(orderedCats);
      availableContainer.querySelectorAll('.cat-item').forEach(el => {
        el.style.display = selectedSet.has(el.dataset.id) ? 'none' : '';
      });
      dropHint.style.display = orderedCats.length ? 'none' : '';
      selectedContainer.querySelectorAll('.selected-item').forEach(el => el.remove());
      orderedCats.forEach(id => {
        const el = document.createElement('div');
        el.className = 'selected-item flex items-center gap-1 px-2 py-1 rounded bg-blue-900 hover:bg-blue-800 text-sm';
        el.draggable = true;
        el.dataset.id = id;
        el.innerHTML = `<span class="flex-1 truncate">${cfg.catNames[id] || id}</span><button type="button" class="text-gray-400 hover:text-white px-1">×</button>`;
        el.querySelector('button').onclick = () => { orderedCats = orderedCats.filter(x => x !== id); updateUI(); save(); };
        selectedContainer.appendChild(el);
      });
      setupSelectedDrag();
    }

    function setupSelectedDrag() {
      selectedContainer.querySelectorAll('.selected-item').forEach(el => {
        el.addEventListener('dragstart', e => { e.dataTransfer.setData('text', el.dataset.id); e.dataTransfer.effectAllowed = 'move'; el.classList.add('opacity-50'); });
        el.addEventListener('dragend', e => { el.classList.remove('opacity-50'); save(); });
        el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('bg-blue-700'); });
        el.addEventListener('dragleave', e => el.classList.remove('bg-blue-700'));
        el.addEventListener('drop', e => {
          e.preventDefault(); el.classList.remove('bg-blue-700');
          const fromId = e.dataTransfer.getData('text');
          const toId = el.dataset.id;
          if (fromId && toId && fromId !== toId) {
            const fromIdx = orderedCats.indexOf(fromId);
            const toIdx = orderedCats.indexOf(toId);
            if (fromIdx !== -1) orderedCats.splice(fromIdx, 1);
            orderedCats.splice(toIdx, 0, fromId);
            updateUI();
          }
        });
      });
    }

    function save() {
      fetch('/settings/guide-filter', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cats: orderedCats})
      });
    }

    availableContainer.querySelectorAll('.cat-item').forEach(el => {
      el.addEventListener('dragstart', e => { e.dataTransfer.setData('text', el.dataset.id); el.classList.add('opacity-50'); });
      el.addEventListener('dragend', e => { el.classList.remove('opacity-50'); save(); });
      el.addEventListener('dblclick', () => { if (!orderedCats.includes(el.dataset.id)) { orderedCats.push(el.dataset.id); updateUI(); save(); } });
    });

    selectedContainer.addEventListener('dragover', e => { e.preventDefault(); selectedContainer.classList.add('border-blue-500'); });
    selectedContainer.addEventListener('dragleave', e => { if (!selectedContainer.contains(e.relatedTarget)) selectedContainer.classList.remove('border-blue-500'); });
    selectedContainer.addEventListener('drop', e => {
      e.preventDefault(); selectedContainer.classList.remove('border-blue-500');
      const id = e.dataTransfer.getData('text');
      if (id && !orderedCats.includes(id)) { orderedCats.push(id); updateUI(); save(); }
    });

    availableContainer.addEventListener('dragover', e => e.preventDefault());
    availableContainer.addEventListener('drop', e => {
      e.preventDefault();
      const id = e.dataTransfer.getData('text');
      if (id && orderedCats.includes(id)) { orderedCats = orderedCats.filter(x => x !== id); updateUI(); save(); }
    });

    searchInput?.addEventListener('input', () => {
      const q = searchInput.value.toLowerCase();
      const selectedSet = new Set(orderedCats);
      availableContainer.querySelectorAll('.cat-item').forEach(el => {
        const match = el.textContent.toLowerCase().includes(q);
        el.style.display = (!match || selectedSet.has(el.dataset.id)) ? 'none' : '';
      });
    });

    updateUI();
  }

  // ============================================================
  // Chrome CC Link Copy
  // ============================================================

  function setupChromeCcLink() {
    const el = document.getElementById('chrome-cc-link');
    if (!el) return;
    el.addEventListener('click', function() {
      const text = 'chrome://settings/captions';
      const orig = el.textContent;
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      el.textContent = 'Copied!';
      setTimeout(() => el.textContent = orig, 1500);
    });
  }

  // ============================================================
  // Caption Style Settings
  // ============================================================

  function setupCaptionStyles() {
    const preview = document.getElementById('cc-preview');
    const selects = document.querySelectorAll('.cc-setting');
    if (!preview || !selects.length) return;

    let ccStyleSettings = cfg.ccStyle || {};

    function hexToRgba(hex, opacity) {
      if (hex === 'transparent') return 'transparent';
      const r = parseInt(hex.slice(1,3), 16);
      const g = parseInt(hex.slice(3,5), 16);
      const b = parseInt(hex.slice(5,7), 16);
      return `rgba(${r},${g},${b},${opacity})`;
    }

    function updatePreview() {
      const s = ccStyleSettings;
      preview.style.color = hexToRgba(s.cc_color || '#ffffff', 1);
      preview.style.textShadow = s.cc_shadow || '0 0 4px black, 0 0 4px black';
      preview.style.backgroundColor = hexToRgba(s.cc_bg || '#000000', s.cc_bg_opacity ?? 0.5);
      preview.style.fontSize = s.cc_size || '1em';
      preview.style.fontFamily = s.cc_font || 'inherit';
    }

    selects.forEach(sel => {
      const key = sel.dataset.setting;
      if (ccStyleSettings[key]) sel.value = ccStyleSettings[key];
      sel.addEventListener('change', function() {
        ccStyleSettings[this.dataset.setting] = this.value;
        fetch('/api/user-prefs', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({cc_style: ccStyleSettings})
        });
        updatePreview();
      });
    });

    updatePreview();
  }

  // ============================================================
  // Caption Language Preference
  // ============================================================

  function setupCaptionLang() {
    const langSelect = document.getElementById('cc-lang-pref');
    if (!langSelect) return;
    if (cfg.ccLang) langSelect.value = cfg.ccLang;
    langSelect.addEventListener('change', function() {
      fetch('/api/user-prefs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cc_lang: this.value})
      });
    });
  }

  // ============================================================
  // Captions Enabled Toggle
  // ============================================================

  function setupCaptionsEnabled() {
    const cb = document.getElementById('captions-enabled');
    if (!cb) return;
    cb.addEventListener('change', () => {
      const form = new FormData();
      if (cb.checked) form.append('enabled', 'on');
      fetch('/settings/captions', { method: 'POST', body: form });
    });
  }

  // ============================================================
  // Transcode Settings
  // ============================================================

  function setupTranscodeSettings() {
    const container = document.getElementById('transcode-settings');
    if (!container) return;

    const defaultBitrates = {'4k': 20, '1080p': 6, '720p': 3, '480p': 1.5};
    const defaultValues = new Set(Object.values(defaultBitrates));
    const bitrateInput = document.getElementById('max-bitrate');

    function saveSettings() {
      const form = new FormData();
      form.append('mode', container.querySelector('input[name="transcode_mode"]:checked')?.value || 'auto');
      form.append('hw', container.querySelector('input[name="transcode_hw"]:checked')?.value || 'nvidia');
      form.append('max_resolution', container.querySelector('input[name="max_resolution"]:checked')?.value || '1080p');
      form.append('max_bitrate_mbps', bitrateInput?.value || '0');
      form.append('vod_transcode_cache_mins', container.querySelector('input[name="vod_transcode_cache_mins"]')?.value || '0');
      if (container.querySelector('input[name="probe_movies"]')?.checked) form.append('probe_movies', 'on');
      if (container.querySelector('input[name="probe_series"]')?.checked) form.append('probe_series', 'on');
      fetch('/settings/transcode', { method: 'POST', body: form });
    }

    container.querySelectorAll('input[name="max_resolution"]').forEach(radio => {
      radio.addEventListener('change', () => {
        const currentVal = parseFloat(bitrateInput?.value) || 0;
        if (!bitrateInput?.value || defaultValues.has(currentVal)) {
          if (bitrateInput) bitrateInput.value = defaultBitrates[radio.value] || '';
          saveSettings();
        }
      });
    });

    container.querySelectorAll('.setting-input').forEach(el => {
      el.addEventListener('change', saveSettings);
    });

    // Re-detect hardware encoders button
    const refreshBtn = document.getElementById('refresh-encoders-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Detecting...';
        try {
          const resp = await fetch('/settings/refresh-encoders', { method: 'POST' });
          if (resp.ok) {
            const data = await resp.json();
            const encoders = data.encoders || {};
            // Update radio button states
            ['nvidia', 'intel', 'vaapi', 'software'].forEach(hw => {
              const radio = container.querySelector(`input[name="transcode_hw"][value="${hw}"]`);
              const label = radio?.closest('label');
              if (radio && label) {
                radio.disabled = !encoders[hw];
                label.classList.toggle('opacity-40', !encoders[hw]);
              }
            });
            refreshBtn.textContent = 'Done!';
            setTimeout(() => { refreshBtn.textContent = 'Re-detect Hardware'; }, 1500);
          } else {
            refreshBtn.textContent = 'Failed';
            setTimeout(() => { refreshBtn.textContent = 'Re-detect Hardware'; }, 1500);
          }
        } catch {
          refreshBtn.textContent = 'Failed';
          setTimeout(() => { refreshBtn.textContent = 'Re-detect Hardware'; }, 1500);
        }
        refreshBtn.disabled = false;
      });
    }
  }

  // ============================================================
  // User-Agent Settings
  // ============================================================

  function setupUserAgentSettings() {
    const container = document.getElementById('user-agent-settings');
    if (!container) return;

    const customContainer = document.getElementById('custom-user-agent-container');
    const presetRadios = container.querySelectorAll('input[name="user_agent_preset"]');
    const customInput = container.querySelector('input[name="user_agent_custom"]');

    function saveSettings() {
      const form = new FormData();
      form.append('preset', container.querySelector('input[name="user_agent_preset"]:checked')?.value || 'default');
      form.append('custom', customInput?.value || '');
      fetch('/settings/user-agent', { method: 'POST', body: form });
    }

    // Toggle custom input visibility
    presetRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        if (radio.value === 'custom') {
          customContainer?.classList.remove('hidden');
        } else {
          customContainer?.classList.add('hidden');
        }
        saveSettings();
      });
    });

    // Save when custom input changes
    customInput?.addEventListener('change', saveSettings);
  }

  // ============================================================
  // Probe Cache Management
  // ============================================================

  function setupProbeCache() {
    const listEl = document.getElementById('probe-cache-list');
    const clearAllBtn = document.getElementById('clear-all-probe-cache');
    if (!listEl) return;

    function escapeHtml(s) {
      if (!s) return '';
      return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatDuration(secs) {
      if (!secs || secs <= 0) return '';
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      if (h > 0) return `${h}h${m}m`;
      return `${m}m`;
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
            const mruEp = s.mru ? episodes.find(ep => ep.episode_id === s.mru) : null;
            const mruName = mruEp ? escapeHtml(mruEp.name) || `Episode ${s.mru}` : null;
            const vcodec = escapeHtml(s.video_codec);
            const acodec = escapeHtml(s.audio_codec);
            return `
              <details class="bg-gray-700 rounded group">
                <summary class="flex items-center justify-between p-2 cursor-pointer hover:bg-gray-600 rounded text-sm">
                  <div class="flex-1 min-w-0">
                    <span class="font-medium truncate">${name}</span>
                    <span class="text-gray-400 ml-2">${s.episode_count} ep${s.episode_count > 1 ? 's' : ''}</span>
                    <span class="text-gray-500 ml-2">${vcodec}/${acodec}</span>
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

          listEl.querySelectorAll('.clear-series').forEach(btn => {
            btn.addEventListener('click', (e) => {
              e.stopPropagation();
              fetch(`/settings/probe-cache/clear/${btn.dataset.series}`, { method: 'POST' }).then(() => loadCache());
            });
          });
          listEl.querySelectorAll('.clear-mru').forEach(btn => {
            btn.addEventListener('click', (e) => {
              e.stopPropagation();
              fetch(`/settings/probe-cache/clear/${btn.dataset.series}`, { method: 'POST' }).then(() => loadCache());
            });
          });
          listEl.querySelectorAll('.clear-episode').forEach(btn => {
            btn.addEventListener('click', (e) => {
              e.stopPropagation();
              fetch(`/settings/probe-cache/clear/${btn.dataset.series}?episode_id=${btn.dataset.episode}`, { method: 'POST' }).then(() => loadCache());
            });
          });
        })
        .catch(() => {
          listEl.innerHTML = '<div class="text-red-400 text-sm">Failed to load</div>';
        });
    }

    clearAllBtn?.addEventListener('click', () => {
      fetch('/settings/probe-cache/clear', { method: 'POST' }).then(() => loadCache());
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
        const refreshType = btn.dataset.refresh;
        btn.classList.add('active');
        activeRefreshes.add(`${sourceId}_${refreshType}`);
        fetch(`/settings/refresh/${sourceId}/${refreshType}`, { method: 'POST' })
          .then(r => r.json())
          .then(() => startPolling())
          .catch(() => btn.classList.remove('active'));
      });
    });

    fetch('/settings/refresh-status')
      .then(r => r.json())
      .then(statuses => {
        if (Object.keys(statuses).length > 0) {
          updateButtonStates(statuses);
          startPolling();
        }
      })
      .catch(() => {});
  }

  // ============================================================
  // User Forms
  // ============================================================

  function setupUserForms() {
    document.getElementById('add-user-form')?.addEventListener('submit', async function(e) {
      e.preventDefault();
      const form = new FormData(this);
      const msgEl = document.getElementById('add-user-msg');
      try {
        const resp = await fetch('/settings/users/add', { method: 'POST', body: form });
        if (resp.ok) {
          msgEl.textContent = 'Added';
          msgEl.className = 'text-sm text-green-400';
          this.reset();
          setTimeout(() => location.reload(), 500);
        } else {
          const data = await resp.json();
          msgEl.textContent = data.detail || 'Failed';
          msgEl.className = 'text-sm text-red-400';
        }
      } catch (err) {
        msgEl.textContent = 'Request failed';
        msgEl.className = 'text-sm text-red-400';
      }
      msgEl.classList.remove('hidden');
      setTimeout(() => { msgEl.className = 'text-sm hidden'; }, 3000);
    });

    // Per-user password forms
    document.querySelectorAll('.password-form').forEach(form => {
      form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const username = this.closest('[data-username]')?.dataset.username;
        if (!username) return;
        const formData = new FormData(this);
        const msgEl = this.querySelector('.password-msg');
        try {
          const resp = await fetch(`/settings/users/password/${username}`, { method: 'POST', body: formData });
          if (resp.ok) {
            msgEl.textContent = 'Saved';
            msgEl.className = 'password-msg text-sm text-green-400';
            this.reset();
          } else {
            const data = await resp.json();
            msgEl.textContent = data.detail || 'Failed';
            msgEl.className = 'password-msg text-sm text-red-400';
          }
        } catch (err) {
          msgEl.textContent = 'Request failed';
          msgEl.className = 'password-msg text-sm text-red-400';
        }
        setTimeout(() => { msgEl.className = 'password-msg text-sm hidden'; }, 3000);
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
          if (!resp.ok) {
            this.checked = !this.checked; // Revert
          } else {
            location.reload(); // Refresh to update badge
          }
        } catch {
          this.checked = !this.checked;
        }
      });
    });
  }

  // ============================================================
  // Init
  // ============================================================

  function init() {
    setupSourceTypeSelect();
    setupCategoryFilter();
    setupChromeCcLink();
    setupCaptionStyles();
    setupCaptionLang();
    setupCaptionsEnabled();
    setupTranscodeSettings();
    setupUserAgentSettings();
    setupProbeCache();
    setupRefreshButtons();
    setupUserForms();
  }

  init();
})();
