# Vendored JavaScript

Served from `/static/vendor/` so the web UI loads without a third-party CDN.
`netv/vendor_test.py` fails if a file stops matching its hash below.
To upgrade, replace the file, update its row, and update the `<script>` tag in
`netv/templates/base.html`.

The files are the unmodified published builds. The minified builds carry no
license header, so each library's license text sits beside them.

| File | Version | Source | SHA-384 (SRI) | License |
|------|---------|--------|---------------|---------|
| `htmx-2.0.4.min.js` | htmx 2.0.4 | https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js | `sha384-HGfztofotfshcF7+8n44JQL2oJmowVChPTg48S+jvZoztPfvwD79OC/LTtG6dMp+` | 0BSD (`htmx-LICENSE`) |
| `hls-1.7.4.min.js` | hls.js 1.7.4 | https://cdn.jsdelivr.net/npm/hls.js@1.7.4/dist/hls.min.js | `sha384-p9vgU/7IYJWd3I4STrtqxqR/2cAQlhIX7D/HrTnIMEjjcUox165b7se7sxmm3zDg` | Apache-2.0 (`hls.js-LICENSE`) |
