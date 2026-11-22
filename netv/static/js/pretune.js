// The guide starts the channel focus rests on, so Enter plays it without the tuner's
// wait: after a short rest, the server pretunes it if a tuner stays free, and leaving
// the guide lets it go. A past programme plays from the archive, so it is never tuned.
(() => {
  const REST_MS = 600;
  const LIVE_LINK = /^\/play\/([a-z2-7]{10})$/;

  window.netvPretune = (root) => {
    let timer = null;
    let wanted = null; // The live link focus rests on.
    let held = null; // The link the server was asked to tune.

    root.addEventListener("focusin", (e) => {
      const link = LIVE_LINK.exec(e.target.getAttribute?.("href") || "")?.[1] || null;
      if (link === wanted) return;
      wanted = link;
      clearTimeout(timer);
      if (!link || link === held) return;
      timer = setTimeout(() => {
        held = link;
        fetch(`/api/pretune/${link}`, { method: "POST" }).catch(() => {});
      }, REST_MS);
    });

    root.addEventListener("focusout", (e) => {
      if (root.contains(e.relatedTarget)) return;
      clearTimeout(timer);
      wanted = null;
      if (held === null) return;
      held = null;
      fetch("/api/pretune", { method: "DELETE" }).catch(() => {});
    });
  };
})();
