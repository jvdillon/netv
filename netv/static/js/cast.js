(function() {
  'use strict';

  window.NetvCast = {
    setup({ serverAddress, getMedia, onStarted, getBrowserMedia, onBrowserStarted }) {
      const el = id => document.getElementById('cast-' + id);
      const dialog = el('dialog');
      let active = false;
      let serverStatus = { active: false };
      let browserSession = null;
      let browserReady = false;
      let statusReady = false;
      let castContext = null;
      let busy = false;
      let timer = null;
      let closed = false;
      let discovering = false;
      let generation = 0;
      el('btn').disabled = true;
      el('server').value = serverAddress || window.location.origin;

      function showError(error) {
        console.warn('[CAST]', error);
        el('error').textContent = error.message || String(error);
        el('error').classList.remove('hidden');
      }

      async function api(path, body) {
        const response = await fetch('/api/cast/' + path, {
          method: body ? 'POST' : 'GET',
          headers: { 'Accept': 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}) },
          ...(body ? { body: JSON.stringify(body) } : {}),
          signal: AbortSignal.timeout(path === 'start' ? 45000 : 15000),
        });
        if (response.redirected) throw new Error('Sign in again to control Chromecast devices.');
        const data = await response.json();
        if (!response.ok) {
          throw new Error(typeof data.detail === 'string' ? data.detail : `Cast request failed (${response.status}).`);
        }
        return data;
      }

      function updateBrowserButton() {
        if (el('browser')) el('browser').disabled = !browserReady || !statusReady || busy || active;
      }

      async function render(status = serverStatus) {
        const wasServerActive = serverStatus.active;
        serverStatus = status;
        const wasActive = active;
        const browserMedia = browserSession && browserSession.getMediaSession();
        const display = browserSession ? {
          active: true,
          name: browserSession.getCastDevice().friendlyName,
          state: browserMedia ? browserMedia.playerState : 'CONNECTED',
          volume: browserSession.getVolume(),
        } : status;
        active = !!(browserSession || status.active);
        updateBrowserButton();
        el('btn').classList.toggle('active', active);
        el('picker').classList.toggle('hidden', active);
        el('remote').classList.toggle('hidden', !active);
        if (active) {
          const text = `Casting to ${display.name} (${display.state.toLowerCase()})`;
          el('now').textContent = text;
          el('state').textContent = text;
          el('volume').value = display.volume;
          el('local').textContent = 'Stop casting and play here';
          el('overlay').classList.remove('hidden');
          if (el('note')) el('note').textContent = browserSession
            ? 'Keep neTV reachable while casting.'
            : 'You can close this page. The TV will keep playing.';
          if (!browserSession && !wasServerActive) await onStarted(status);
        } else if (wasActive) {
          el('state').textContent = 'Casting has ended.';
          el('local').textContent = 'Play here';
        }
      }

      async function poll() {
        const pollGeneration = generation;
        try {
          if (!busy) {
            const status = await api('status');
            if (!busy && generation === pollGeneration && !closed) await render(status);
          }
        } catch (error) {
          showError(error);
          if (serverStatus.active && !browserSession) el('state').textContent = 'Cannot reach neTV. The TV may still be playing.';
        } finally {
          if (!closed) timer = setTimeout(poll, 3000);
        }
      }

      async function discover() {
        if (discovering) return;
        discovering = true;
        el('refresh').disabled = true;
        el('discovery').textContent = 'Searching the server’s network...';
        el('devices').replaceChildren();
        try {
          const data = await api('devices');
          el('discovery').textContent = data.devices.length
            ? 'Select a TV, or enter its IP below.'
            : 'No devices found. Enter the TV IP below; Docker or guest Wi-Fi may block discovery.';
          for (const device of data.devices) {
            const button = document.createElement('button');
            button.className = 'px-3 py-2 bg-gray-700 rounded text-left';
            button.textContent = `${device.name} (${device.host})`;
            button.addEventListener('click', () => {
              el('ip').value = device.host;
              start();
            });
            el('devices').appendChild(button);
          }
        } catch (error) {
          el('discovery').textContent = 'Discovery unavailable. Try entering the TV IP manually.';
          showError(error);
        } finally {
          discovering = false;
          el('refresh').disabled = false;
        }
      }

      async function operation(action) {
        if (busy) return false;
        busy = true;
        generation++;
        el('error').classList.add('hidden');
        const controls = ['start', 'play', 'pause', 'stop', 'volume', 'local'];
        controls.forEach(id => { el(id).disabled = true; });
        try {
          await action();
          return true;
        } catch (error) {
          showError(error);
          if (!dialog.open) dialog.showModal();
          return false;
        } finally {
          busy = false;
          controls.forEach(id => { el(id).disabled = false; });
          updateBrowserButton();
        }
      }

      async function start() {
        if (browserSession || active) return;
        await operation(async () => {
          el('discovery').textContent = 'Preparing stream and connecting to the TV...';
          const media = await getMedia();
          const status = await api('start', {
            ...media, host: el('ip').value.trim(), server_url: el('server').value.trim(),
          });
          await render(status);
        });
      }

      function browserUnavailable(message) {
        browserReady = false;
        if (el('browser-status')) el('browser-status').textContent = message;
        updateBrowserButton();
      }

      function resumeBrowserSession() {
        const session = castContext.getCurrentSession();
        if (!statusReady || !session || !session.getMediaSession() || serverStatus.active || busy || browserSession === session) return;
        const media = session.getMediaSession().media;
        if (!media || !media.contentId) return;
        let url;
        try { url = new URL(media.contentId); } catch { return; }
        if (url.origin !== window.location.origin || !url.pathname.startsWith('/transcode/')) return;
        browserSession = session;
        render().catch(showError);
        if (onBrowserStarted) Promise.resolve(onBrowserStarted()).catch(showError);
      }

      function initializeBrowser(available) {
        if (!available || !window.cast || !window.chrome || !window.chrome.cast) {
          browserUnavailable('Nearby casting is unavailable in this browser. Use server network casting below.');
          return;
        }
        try {
          const framework = window.cast.framework;
          castContext = framework.CastContext.getInstance();
          castContext.setOptions({
            receiverApplicationId: window.chrome.cast.media.DEFAULT_MEDIA_RECEIVER_APP_ID,
            autoJoinPolicy: window.chrome.cast.AutoJoinPolicy.ORIGIN_SCOPED,
          });
          castContext.addEventListener(framework.CastContextEventType.SESSION_STATE_CHANGED, event => {
            if (event.sessionState === framework.SessionState.SESSION_ENDED) {
              browserSession = null;
              render().catch(showError);
            } else {
              resumeBrowserSession();
            }
          });
          browserReady = true;
          el('browser-status').textContent = 'Choose a nearby Cast device on your browser’s network.';
          updateBrowserButton();
          resumeBrowserSession();
        } catch (error) {
          browserUnavailable('Nearby casting could not initialize. Use server network casting below.');
        }
      }

      function startBrowser() {
        if (!browserReady || !statusReady || busy || active) return;
        // Request the picker before any asynchronous media preparation loses user activation.
        let requested;
        try {
          requested = castContext.getCurrentSession() ? Promise.resolve() : castContext.requestSession();
        } catch (error) {
          showError(error);
          return;
        }
        return operation(async () => {
          updateBrowserButton();
          let session;
          try {
            await requested;
            session = castContext.getCurrentSession();
            if (!session) throw new Error('No nearby Cast session was selected.');
            const media = await getBrowserMedia();
            const sdk = window.chrome.cast.media;
            const info = new sdk.MediaInfo(media.url, 'application/x-mpegURL');
            info.streamType = media.isVod ? sdk.StreamType.BUFFERED : sdk.StreamType.LIVE;
            info.hlsSegmentFormat = 'ts';
            info.hlsVideoSegmentFormat = 'mpeg2_ts';
            info.metadata = new sdk.GenericMediaMetadata();
            info.metadata.title = media.title;
            const request = new sdk.LoadRequest(info);
            request.autoplay = true;
            if (media.isVod) request.currentTime = media.current_time || 0;
            await session.loadMedia(request);
            browserSession = session;
            await render();
            if (onBrowserStarted) await onBrowserStarted();
          } catch (error) {
            if (session && !browserSession) castContext.endCurrentSession(true);
            throw error;
          }
        });
      }

      async function controlBrowser(action) {
        if (action === 'stop') {
          await castContext.endCurrentSession(true);
          browserSession = null;
        } else if (action === 'volume') {
          await browserSession.setVolume(Number(el('volume').value));
        } else {
          const media = browserSession.getMediaSession();
          if (!media) throw new Error('No media is playing on the nearby device.');
          await new Promise((resolve, reject) => media[action](null, resolve, reject));
        }
        await render();
      }

      if (el('browser')) {
        el('browser').disabled = true;
        el('browser').addEventListener('click', startBrowser);
        const location = window.location;
        const secure = location.protocol === 'https:' ||
          ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
        if (!getBrowserMedia) {
          browserUnavailable('Nearby casting is not available for this media. Use server network casting below.');
        } else if (!secure) {
          browserUnavailable('Nearby casting requires HTTPS or localhost. Use server network casting below.');
        } else {
          el('browser-status').textContent = 'Loading nearby casting support…';
          window.__onGCastApiAvailable = initializeBrowser;
          if (window.cast && window.cast.framework) initializeBrowser(true);
          else {
            const script = document.createElement('script');
            script.src = 'https://www.gstatic.com/cv/js/sender/v1/cast_sender.js?loadCastFramework=1';
            script.onerror = () => initializeBrowser(false);
            document.head.appendChild(script);
          }
        }
      }

      async function open() {
        if (!dialog.open) dialog.showModal();
        if (!active) await discover();
      }

      el('btn').addEventListener('click', event => { event.stopPropagation(); open(); });
      el('manage').addEventListener('click', open);
      el('close').addEventListener('click', () => dialog.close());
      el('refresh').addEventListener('click', discover);
      el('form').addEventListener('submit', event => { event.preventDefault(); start(); });
      for (const action of ['play', 'pause', 'stop']) {
        el(action).addEventListener('click', () => operation(async () => {
          if (browserSession) await controlBrowser(action);
          else await render(await api('control', { action }));
        }));
      }
      el('volume').addEventListener('change', () => operation(async () => {
        if (browserSession) await controlBrowser('volume');
        else await render(await api('control', { action: 'volume', volume: Number(el('volume').value) }));
      }));
      el('local').addEventListener('click', async () => {
        const stopped = await operation(async () => {
          if (browserSession) await controlBrowser('stop');
          else if (serverStatus.active) await render(await api('control', { action: 'stop' }));
        });
        if (stopped) window.location.reload();
      });
      window.addEventListener('pagehide', () => { closed = true; clearTimeout(timer); });
      window.addEventListener('pageshow', event => {
        if (event.persisted) { closed = false; poll(); }
      });
      const ready = poll().finally(() => {
        statusReady = true;
        el('btn').disabled = false;
        updateBrowserButton();
        if (castContext) resumeBrowserSession();
      });
      return {
        ready, get active() { return active; }, get browserActive() { return !!browserSession; },
        get busy() { return busy; }, open,
      };
    },
  };
})();
