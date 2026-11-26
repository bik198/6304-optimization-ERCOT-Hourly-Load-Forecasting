# Beadline Architecture & Frontend Playbook (`memory.md`)

> **Purpose:** Portable blueprint of the **Beadline** interactive presentation + necklace explorer in `optiFinal` (`ASDS 6304`), designed so a second project—specifically the **Time Series Final (`ASDS 5306 · Group 13`)**—reuses `web/` unchanged with only a new `beadline_project.py` config and Python exporters.

---

## 1. Non-Negotiable Content & Design Rules

1. **Facts Only — Every Claim Must Be Sourced & Computed in Python**
   - Never type unverified numbers, unmeasured causal explanations, editorial adjectives (*"honest"*, *"crucially"*, *"for free"*, *"cheating"*), or hand-calculated ratios in story text.
   - Compute every ratio, difference, percentage, min/max, and win rate inside `facts()` in `export_story_data.py` so the Overview and every Scene pull from the exact same Python variables.
2. **Read Mode vs. Present Mode (`F`) — Keep Evidence Above the Fold**
   - **Read Mode (default):** Sidebar visible + full scene summary paragraph (`stop.sub`) + stats + chart + callouts + expandable `Proof & Technical Details` drawer (`stop.details`).
   - **Present Mode (`F` key):** Hides the sidebar, moves `stop.sub` into `Notes` / Presenter View, caps the headline at 2 lines, compacts stat tiles, and fits the primary chart **100% above the fold** on `1520×940` / `1280×800` projectors.
   - **Click-to-Reveal (`1 → 2 → 3`):** In Present Mode, pressing `→`, `Space`, or clicking the stage reveals `1 · Stats` $\to$ `2 · Chart` $\to$ `3 · Callout` before advancing to the next scene.
3. **Second-Window Presenter Console (`P` or `?presenter=1`)**
   - Synced to the projected tab via `BroadcastChannel('beadline-sync')`.
   - Displays a live elapsed timer (`Pause` / `Reset`) + wall clock, `← Step Back` / `Advance →` projector controls, large **`Say out loud`** cue, the hidden `stop.sub` summary, editable `localStorage` speaker notes, and an **`Up Next`** preview of the next scene.
4. **True Necklace Machine View (`M`)**
   - Pixel-true SVG (`ResizeObserver`, no `viewBox` shrink-scaling, real `12px` labels).
   - Each strand renders as **one straight horizontal string of beads** across the full height of the viewport, using `color-mix(in srgb, strandColor 78%, var(--text-bright))` so pale hues remain crisp in both Dark and Light themes.
   - Cross-strand dependencies route through **orthogonal rounded-corner trunks** that merge visually into bundled cables instead of diagonal spaghetti.
   - Labels appear only on **hover or select** (in floating pills); Part-colored scene badges always show; arriving via `See in Machine →` highlights all `stop.beads` for that scene and dims the rest.
5. **Chart Hygiene**
   - Models or solvers contrasted on the same plot must never share a hue family.
   - `LineChart` reserves a right-hand label gutter (`m.r = 152`) so end-of-line labels always sit to the right of the curve (`textAnchor="start"`) and never collide with plotted lines.
   - Busy multi-model charts mark only 2–3 key series as primary and pass `secondary: true` on remaining context series (rendered as quiet grey lines until hovered in the legend or chart).
   - Integer share bars (`exactPercents`) use largest-remainder rounding so percentages always sum to exact `100%`.
6. **Projector-Ready Light Mode + Offline USB Build + PDF Fallback**
   - Projectors wash out dark themes: always verify every scene and the Machine view in Light mode (`T` or `?theme=light`).
   - `npm run offline` bundles Vite's JS, CSS, base64 woff2 fonts, images, and `beads.json` / `story.json` / `eda.json` (`window.__BEADLINE_INLINE_DATA__`) into **one standalone `web/dist-offline/beadline-offline.html`** that opens directly from `file:///` off a USB stick.
   - `npm run pdf` (or `?print=1`) renders the Overview + all 8 scenes 1-per-landscape-page into `web/dist-offline/ercot-8-scenes.pdf`.

---

## 2. Reusing `web/` for the `ASDS 5306` Time Series Final

The React viewer in `web/` has **zero hardcoded project titles, part names, part colours, or model names**. To stand up the Time Series Final:

1. **Copy `web/` as-is** (keep `App.jsx`, `StopView.jsx`, `Overview.jsx`, `StoryRail.jsx`, `PresenterView.jsx`, `PrintDeck.jsx`, `charts.jsx`, `MachineCanvas.jsx`, `CodeLayer.jsx`, `NumbersLayer.jsx`, `BinaryLayer.jsx`, `ieee754.js`, `styles.css`, and `scripts/bundle_offline.mjs` + `scripts/export_pdf.mjs`).
2. **Copy `src/beads.py`** and list your Time Series strand modules in `STRAND_MODULES`.
3. **Create a new `beadline_project.py`** in the Time Series repo defining:
   - `PROJECT`: `brand`, `docTitle`, `title`, `subtitle`, `byline`, `defaultBead`, `source`
   - `PARTS`: 4 parts (`id`, `title`, `color`, `question`)
   - `MODEL`: model key $\to$ `(display_label, series_color_index_0_to_7)`
   - `STRAND_META`: strand `id`, `label`, `color`, `summary`
4. **Write 3 deterministic exporters** (`export_web_data.py` $\to$ `eda.json`, `export_beads_data.py` $\to$ `beads.json`, `export_story_data.py` $\to$ `story.json`) following the schemas in `optiFinal/export_story_data.py` (8 scenes with `title` $\le$ 22 chars, `beads: [...]` per scene, `say`, `stats`, `charts`, `callouts`, and `details` proof drawers).
5. *(Optional)* Add 2–3 SVG chart cases in `web/src/story/charts.jsx` if needed for `eda:stl` (STL decomposition), `eda:acf` (ACF/PACF stem plots with $\pm 1.96/\sqrt{T}$ bands), or `eda:fan` (80%/95% forecast interval fan charts).

---

## 3. Build, Offline Bundle & Verification Commands (Windows PowerShell)

```powershell
# 1. Regenerate all JSON data from Python strands & results
python export_beads_data.py; python export_story_data.py

# 2. Build web app + standalone USB single-file HTML + landscape PDF fallback
Push-Location web
npm run offline
npm run pdf
Pop-Location
```

- **Interactive dev server:** `cd web; npm run dev`
- **USB single-file fallback:** `web/dist-offline/beadline-offline.html`
- **Landscape PDF fallback:** `web/dist-offline/ercot-8-scenes.pdf`
- **Keyboard shortcuts:** `←` / `→` / `Space` (step/scene) · `F` (Present mode + click-to-reveal) · `P` (synced Presenter console) · `M` (Machine necklace view) · `↓` / `↑` (Code / Numbers / Bits zoom) · `N` (Notes) · `T` (Light / Dark) · `?print=1` (Print/PDF view).
