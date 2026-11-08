// Continue watching: offers the channel left playing on another device.
(function() {
  'use strict';

  const card = document.getElementById('handoff-card');
  if (!card) return;
  const resume = document.getElementById('handoff-resume');

  function clock(seconds) {
    return new Date(seconds * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  function render(handoff) {
    if (!handoff) {
      card.classList.add('hidden');
      return;
    }
    document.getElementById('handoff-title').textContent =
      handoff.program_title ? `${handoff.program_title} · ${handoff.channel_name}` : handoff.channel_name;
    const behind = Math.max(0, Math.round((Date.now() / 1000 - handoff.position) / 60));
    const detail = [`Left at ${clock(handoff.position)}`];
    if (behind >= 1) detail.push(`${behind} min behind live`);
    if (handoff.program_end > Date.now() / 1000) detail.push(`ends ${clock(handoff.program_end)}`);
    document.getElementById('handoff-detail').textContent = detail.join(' · ');
    resume.href = '/play/live/' + encodeURIComponent(handoff.stream_id) +
      '?handoff=' + encodeURIComponent(handoff.id);
    card.classList.remove('hidden');
  }

  async function refresh() {
    try {
      const response = await fetch('/api/handoff', { cache: 'no-store' });
      if (response.ok) render((await response.json()).handoff);
    } catch (e) {
      console.warn('[HANDOFF] Could not check for a handoff:', e);
    }
  }

  document.getElementById('handoff-dismiss').addEventListener('click', async () => {
    card.classList.add('hidden');
    try {
      await fetch('/api/handoff', { method: 'DELETE' });
    } catch (e) {
      console.warn('[HANDOFF] Could not dismiss the handoff:', e);
    }
  });

  refresh();
  // A stop sent as the player closed may still be setting up its recording.
  setTimeout(refresh, 4000);
  // Coming back to the guide (including from the back-forward cache) may follow a stop.
  window.addEventListener('pageshow', event => { if (event.persisted) refresh(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
})();
