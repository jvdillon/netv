(function() {
  'use strict';

  const video = document.getElementById('video');
  const badge = document.getElementById('quality-badge');
  if (!video || !badge) return;

  function hide() {
    badge.classList.add('hidden');
    badge.textContent = '';
    badge.removeAttribute('aria-label');
  }

  function update() {
    const { videoWidth: width, videoHeight: height } = video;
    if (!width || !height) {
      hide();
      return;
    }
    // Match the Apple client: letterboxed 1920x800 still counts as 1080p.
    const resolution = Math.max(height, width * 9 / 16);
    const label = resolution >= 2000 ? '4K'
      : resolution >= 1300 ? '1440p'
      : resolution >= 900 ? '1080p'
      : resolution >= 650 ? '720p'
      : resolution >= 520 ? '576p'
      : resolution >= 400 ? '480p'
      : 'SD';
    badge.textContent = label;
    badge.setAttribute('aria-label', `Video resolution: ${label}`);
    badge.classList.remove('hidden');
  }

  for (const event of ['loadedmetadata', 'resize', 'playing']) {
    video.addEventListener(event, update);
  }
  for (const event of ['loadstart', 'emptied', 'error']) {
    video.addEventListener(event, hide);
  }
  update();
})();
