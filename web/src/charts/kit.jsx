// Shared chart pieces: sizing, scales, ticks, formatting, tooltip, and the Figure card.
// Charts are plain SVG. Spec (dataviz method): 2px lines, ≤24px bars with 4px rounded data
// ends, hairline solid gridlines, ≥8px markers with a 2px surface ring, text in ink tokens
// (never in series colour), a tooltip on every chart and a table view on every figure.
import { useEffect, useLayoutEffect, useRef, useState } from 'react';

export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'];
// the entity → colour map is fixed across the whole story (colour follows the entity)
export const SPLIT_COLOR = { train: SERIES[0], val: SERIES[1], test: SERIES[2] };
export const SPLIT_LABEL = { train: 'Train', val: 'Validation', test: 'Test' };
export const SEASON_COLOR = { winter: SERIES[0], summer: SERIES[1], spring: SERIES[2], fall: SERIES[3] };
export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Current theme name ('light' | 'semi' | 'dark'), from data-theme on <html>; re-renders on change. */
export const getTheme = () => document.documentElement.dataset.theme || 'light';
export function useTheme() {
  const [theme, setTheme] = useState(getTheme);
  useEffect(() => {
    const mo = new MutationObserver(() => setTheme(getTheme()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);
  return theme;
}

/** Width of a container, kept current with ResizeObserver. */
export function useWidth(initial = 800) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export const scale = (d0, d1, r0, r1) => {
  const k = (r1 - r0) / (d1 - d0 || 1);
  const f = v => r0 + (v - d0) * k;
  f.invert = px => d0 + (px - r0) / k;
  return f;
};

/** Round tick values ("nice" steps of 1, 2, 2.5, 5 × 10^n) that always enclose [min, max],
 *  so the first and last tick can serve as the scale's domain. */
export function niceTicks(min, max, count = 5) {
  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
  const start = Math.floor(min / step + 1e-9) * step;
  const end = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 1e-9; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

export const fmtInt = v => Math.round(v).toLocaleString('en-US');
export const fmtMW = v => `${fmtInt(v)} MW`;
export const fmtGW = (v, d = 1) => `${(v / 1000).toFixed(d)} GW`;
export const fmtK = v => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}K` : `${v}`);
export const fmtPct = (v, d = 1) => `${v > 0 ? '+' : ''}${v.toFixed(d)}%`;
export const hourLabel = h => `${String(h).padStart(2, '0')}:00`;
export const fmtDate = (s, opts = { month: 'short', day: 'numeric', year: 'numeric' }) =>
  new Date(`${s.slice(0, 10)}T00:00:00`).toLocaleDateString('en-US', opts);

/** Horizontal gridlines + y tick labels (left), hairline and recessive. */
export function YGrid({ ticks, y, x0, x1, format = fmtK, unit }) {
  return (
    <g>
      {ticks.map(t => (
        <g key={t}>
          <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth="1" />
          <text className="tick" x={x0 - 8} y={y(t)} dy="0.32em" textAnchor="end">{format(t)}</text>
        </g>
      ))}
      {unit && <text className="tick" x={x0 - 8} y={y(ticks[ticks.length - 1]) - 16} textAnchor="end">{unit}</text>}
    </g>
  );
}

/** Tooltip box positioned near (x, y) inside the chart, flipped to stay inside. */
export function Tooltip({ x, y, width, children }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 160, h: 60 });
  useLayoutEffect(() => {
    if (ref.current) setSize({ w: ref.current.offsetWidth, h: ref.current.offsetHeight });
  });
  const left = x + 14 + size.w > width ? x - 14 - size.w : x + 14;
  const top = Math.max(0, y - size.h - 10);
  return <div ref={ref} className="tooltip" style={{ left: Math.max(0, left), top }}>{children}</div>;
}

export function TipRow({ color, label, value }) {
  return <div className="t-row"><i style={{ background: color }} /><span>{label}</span><span>{value}</span></div>;
}

/** The white chart card: title, subtitle, optional controls/legend, chart, then a footer with source and a table toggle. */
export function Figure({ title, sub, controls, legend, table, source = 'ERCOT hourly system load, 2018-01-01 → 2026-11-04', children }) {
  const [showTable, setShowTable] = useState(false);
  return (
    <figure className="figure" style={{ margin: 0 }}>
      <div className="figure-head">
        <div>
          <div className="figure-title">{title}</div>
          {sub && <div className="figure-sub">{sub}</div>}
        </div>
        {controls}
      </div>
      {legend}
      {showTable && table ? <div className="table-wrap">{table}</div> : children}
      <figcaption className="figure-foot">
        <span className="label">{source}</span>
        {table && (
          <button type="button" className="link-btn" onClick={() => setShowTable(s => !s)} aria-pressed={showTable}>
            {showTable ? 'Show chart' : 'Show the numbers'}
          </button>
        )}
      </figcaption>
    </figure>
  );
}

export function DataTable({ columns, rows }) {
  return (
    <table className="data">
      <thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody>
    </table>
  );
}

export function Legend({ items, box = false }) {
  return (
    <div className="legend">
      {items.map(it => <span key={it.label}><i className={box ? 'box' : ''} style={{ background: it.color }} />{it.label}</span>)}
    </div>
  );
}

/** Column path: square at the baseline, 4px rounded data end. */
export function columnPath(x, yTop, w, yBase, r = 4) {
  const rr = Math.min(r, w / 2, Math.max(0, yBase - yTop));
  return `M${x},${yBase}V${yTop + rr}Q${x},${yTop} ${x + rr},${yTop}H${x + w - rr}Q${x + w},${yTop} ${x + w},${yTop + rr}V${yBase}Z`;
}
/** Horizontal bar path: square at the left baseline, 4px rounded right end. */
export function barPath(x0, y, x1, h, r = 4) {
  const rr = Math.min(r, h / 2, Math.max(0, x1 - x0));
  return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 - rr},${y + h}H${x0}Z`;
}

export const linePath = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
