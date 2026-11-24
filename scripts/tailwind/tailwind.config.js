// Config for scripts/build-tailwind.sh. Paths are relative to the repo root.
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "netv/templates/**/*.html",
    "netv/static/js/**/*.js",
    // Python is not scanned: nothing under netv/*.py builds HTML class
    // strings, and scanning it would pull in rules for ordinary words
    // ("container", "fixed", "table"). Add "netv/**/*.py" if that changes.
  ],
  // Classes that are only ever assembled at runtime would go here. The audit
  // of every className/classList/template-literal site found none: all class
  // names in the JS and in Jinja conditionals (for example
  // virtual-guide.js:681, settings.js:20) are whole literal tokens, which the
  // content scan finds. If you build a class by concatenation, for example
  // `"bg-" + color`, list the concrete classes here with a comment naming the
  // source line.
  safelist: [],
  theme: {},
  plugins: [],
};
