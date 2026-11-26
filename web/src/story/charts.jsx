// Story charts: generic line chart (linear or log axes, with secondary context lines & right-gutter labels)
// and ranked bar / dot chart, plus Step 0 EDA charts.
import { useMemo, useState } from 'react';
import {
  DataTable, Figure, SERIES, Tooltip, TipRow, barPath, fmtInt, hourLabel, linePath, niceTicks, useWidth,
} from '../charts/kit';
import { GrowthTimeline } from '../charts/growth';
import { HourProfile } from '../charts/daily';
import { UriPanels } from '../charts/weather';
import { SplitRange } from '../charts/splits';

const INK = 'var(--bright)';
const colorOf = i => (i == null ? INK : SERIES[i]);
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = e => `10${String(e).split('').map(c => SUP[c]).join('')}`;
const fmtNum = v => (Math.abs(v) >= 100 ? fmtInt(v) : Math.abs(v) >= 1 ? v.toLocaleString('en-US', { maximumFractionDigits: 2 }) : v.toPrecision(3));
const DEFAULT_SOURCE = 'Saved results · optimization/results, preliminary';

function axis({ log, zero, exact }, values, r0, r1, count = 5) {
  if (log) {
    const lo = Math.floor(Math.log10(Math.min(...values)));
    const hi = Math.ceil(Math.log10(Math.max(...values)));
    const every = Math.max(1, Math.ceil((hi - lo) / 8));
    const ticks = [];
    for (let e = lo; e <= hi; e += every) ticks.push(10 ** e);
    const k = (r1 - r0) / (hi - lo || 1);
    return { f: v => r0 + (Math.log10(v) - lo) * k, ticks, format: v => pow10(Math.round(Math.log10(v))) };
  }
  const min = zero ? 0 : Math.min(...values);
  const ticks = niceTicks(min, Math.max(...values), count);
  const d0 = exact ? Math.min(...values) : ticks[0], d1 = exact ? Math.max(...values) : ticks[ticks.length - 1];
  const k = (r1 - r0) / (d1 - d0 || 1);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  const format = v => (step >= 1000 ? `${v / 1000}K` : v.toLocaleString('en-US'));
  return { f: v => r0 + (v - d0) * k, ticks, format };
}

/** Interactive legend supporting primary series + hover-to-reveal secondary context series. */
function InteractiveLegend({ series, focusSeries, onFocusSeries }) {
  return (
    <div className="legend">
      {series.map(s => {
        const active = focusSeries === s.label;
        const muted = s.secondary && !active;
        return (
          <span
            key={s.label}
            className={`legend-item ${s.secondary ? 'legend-sec' : ''} ${active ? 'legend-active' : ''}`}
            onMouseEnter={() => onFocusSeries(s.label)}
            onMouseLeave={() => onFocusSeries(null)}
          >
            <i style={{ background: muted ? 'var(--line-strong)' : colorOf(s.color), opacity: muted ? 0.65 : 1 }} />
            {s.label}
          </span>
        );
      })}
    </div>
  );
}

export function LineChart({ spec, compact = false, source = DEFAULT_SOURCE }) {
  const [ref, width] = useWidth();
  const [hoverX, setHoverX] = useState(null);
  const [focusSeries, setFocusSeries] = useState(null);
  const { series } = spec;

  const shownLabels = useMemo(
    () => series.filter(s => !s.secondary || s.label === focusSeries),
    [series, focusSeries],
  );
  const direct = shownLabels.length > 1 && shownLabels.length <= 4 && !spec.x.hourly;
  const H = compact ? 268 : 318;
  const m = { l: 58, r: direct ? 152 : 26, t: 22, b: 38 };

  const geo = useMemo(() => {
    const xs = series.flatMap(s => s.points.map(p => p[0]));
    const ys = series.flatMap(s => s.points.map(p => p[1]));
    const x = axis({ ...spec.x, exact: !!spec.x.ticks }, xs, m.l, Math.max(m.l + 80, width - m.r), 6);
    if (spec.x.hours) x.ticks = [0, 3, 6, 9, 12, 15, 18, 21, 23];
    if (spec.x.ticks) x.ticks = spec.x.ticks.map(t => t.v);
    const y = axis(spec.y, ys, H - m.b, m.t);
    return { x, y };
  }, [series, spec, width, H, m.l, m.r, m.t, m.b]);
  const { x, y } = geo;

  const onMove = e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    setHoverX(px >= m.l - 4 && px <= width - m.r + 8 ? px : null);
  };
  const nearest = s => s.points.reduce((best, p) => (Math.abs(x.f(p[0]) - hoverX) < Math.abs(x.f(best[0]) - hoverX) ? p : best), s.points[0]);
  const hoverPts = hoverX == null ? null : series.map(s => ({ s, p: nearest(s) }));
  const tickLabel = v => {
    const ticks = spec.x.ticks || [];
    const base = [...ticks].reverse().find(t => t.v <= v);
    if (!base) return fmtNum(v);
    return spec.x.hourly ? `${base.label} ${hourLabel(v - base.v)}` : base.label;
  };
  const xFmt = v => (spec.x.hours ? hourLabel(v) : spec.x.ticks ? tickLabel(v) : fmtNum(v));

  // De-overlapped end-of-line y positions for visible direct labels (always placed to the right of the line end)
  const labelY = useMemo(() => {
    const ends = shownLabels
      .map(s => ({ label: s.label, y: y.f(s.points[s.points.length - 1][1]) }))
      .sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 15);
    if (ends.length > 0 && ends[ends.length - 1].y > H - m.b - 2) {
      ends[ends.length - 1].y = H - m.b - 2;
      for (let k = ends.length - 2; k >= 0; k--) ends[k].y = Math.min(ends[k].y, ends[k + 1].y - 15);
    }
    return new Map(ends.map(e => [e.label, e.y]));
  }, [shownLabels, y, H, m.b]);

  // Draw secondary context lines first so primary curves sit crisply on top
  const orderedSeries = useMemo(
    () => [...series.filter(s => s.secondary && s.label !== focusSeries), ...series.filter(s => !s.secondary || s.label === focusSeries)],
    [series, focusSeries],
  );

  return (
    <Figure
      title={spec.title}
      source={source}
      legend={series.length > 1 ? <InteractiveLegend series={series} focusSeries={focusSeries} onFocusSeries={setFocusSeries} /> : null}
      table={<DataTable columns={['Series', spec.x.label, `${spec.y.label}${spec.y.unit ? ` (${spec.y.unit})` : ''}`]}
        rows={series.flatMap(s => s.points.map(p => [s.label, xFmt(p[0]), fmtNum(p[1])]))} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={spec.title} onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}>
          {y.ticks.map(t => (
            <g key={t}>
              <line x1={m.l} x2={width - m.r} y1={y.f(t)} y2={y.f(t)} stroke="var(--grid)" />
              <text className="tick" x={m.l - 8} y={y.f(t)} dy="0.32em" textAnchor="end">{y.format(t)}</text>
            </g>
          ))}
          {spec.y.unit && <text className="tick" x={m.l - 8} y={m.t - 9} textAnchor="end">{spec.y.unit}</text>}
          {x.ticks.map(t => (
            <text key={t} className="tick" x={x.f(t)} y={H - m.b + 17} textAnchor="middle">
              {spec.x.hours ? String(t).padStart(2, '0') : spec.x.ticks ? spec.x.ticks.find(k => k.v === t)?.label : x.format(t)}
            </text>
          ))}
          <text className="tick" x={width - m.r} y={H - 4} textAnchor="end">{spec.x.label}{spec.x.unit ? ` (${spec.x.unit})` : ''}</text>
          {orderedSeries.map(s => {
            const pts = s.points.map(p => [x.f(p[0]), y.f(p[1])]);
            const isFocused = focusSeries === s.label;
            const isMutedSec = s.secondary && !isFocused;
            const dimByFocus = focusSeries && !isFocused && !s.secondary;
            const stroke = isMutedSec ? 'var(--line-strong)' : colorOf(s.color);
            const sw = isFocused ? 2.6 : isMutedSec ? 1.25 : 2.1;
            const op = isMutedSec ? (hoverX != null ? 0.65 : 0.4) : dimByFocus ? 0.45 : 1;
            const endPt = pts[pts.length - 1];
            const ly = labelY.get(s.label);
            return (
              <g key={s.label}>
                <path d={linePath(pts)} fill="none" stroke={stroke} strokeWidth={sw} strokeOpacity={op} strokeLinejoin="round" />
                {s.markers && !isMutedSec && pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="4" fill={stroke} stroke="var(--surface)" strokeWidth="2" />)}
                {direct && ly != null && (
                  <text className="ann chart-end-label" x={endPt[0] + 7} y={ly} dy="0.32em" textAnchor="start">
                    {s.label}
                  </text>
                )}
              </g>
            );
          })}
          {hoverPts && (
            <g pointerEvents="none">
              <line x1={hoverX} x2={hoverX} y1={m.t} y2={H - m.b} stroke="var(--line-strong)" />
              {hoverPts.map(({ s, p }) => (
                <circle
                  key={s.label}
                  cx={x.f(p[0])}
                  cy={y.f(p[1])}
                  r={s.secondary && focusSeries !== s.label ? 3.2 : 4.5}
                  fill={colorOf(s.color)}
                  stroke="var(--surface)"
                  strokeWidth="2"
                />
              ))}
            </g>
          )}
        </svg>
        {hoverPts && (
          <Tooltip x={hoverX} y={m.t + 36} width={width}>
            <div className="t-head">{xFmt(hoverPts[0].p[0])}</div>
            {hoverPts.map(({ s, p }) => (
              <TipRow key={s.label} color={colorOf(s.color)} label={s.label} value={`${fmtNum(p[1])}${spec.y.unit ? ` ${spec.y.unit}` : ''}`} />
            ))}
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

/** Ranked rows: bars from zero ("bars") or dots on a zoomed axis ("dots"). */
export function RankChart({ spec, dots = false, compact = false, source = DEFAULT_SOURCE }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const rows = spec.rows;
  const rowH = compact ? 26 : 30;
  const labelW = 214, m = { t: 8, b: 28, r: 72 };
  const H = m.t + rows.length * rowH + m.b;
  const values = rows.map(r => r.value);
  const x = axis({ zero: !dots }, values, labelW, Math.max(labelW + 80, width - m.r), 5);
  const color = colorOf(spec.color);

  return (
    <Figure
      title={spec.title}
      source={source}
      table={<DataTable columns={['Setting', `Value (${spec.unit})`]} rows={rows.map(r => [r.label, fmtNum(r.value)])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={spec.title} onMouseLeave={() => setHover(null)}>
          {x.ticks.map(t => (
            <g key={t}>
              <line x1={x.f(t)} x2={x.f(t)} y1={m.t} y2={H - m.b} stroke="var(--grid)" />
              <text className="tick" x={x.f(t)} y={H - m.b + 17} textAnchor="middle">{fmtInt(t)}</text>
            </g>
          ))}
          <text className="tick" x={width - 8} y={H - m.b + 17} textAnchor="end">{spec.unit}</text>
          {rows.map((r, i) => {
            const cy = m.t + i * rowH + rowH / 2;
            const fill = r.highlight ? color : 'var(--muted)';
            return (
              <g key={r.label} onMouseEnter={() => setHover(i)}>
                <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill={hover === i ? 'var(--hover)' : 'transparent'} />
                <text className={r.highlight ? 'ann' : 'tick'} x={labelW - 12} y={cy} dy="0.32em" textAnchor="end">{r.label}</text>
                {dots ? (
                  <>
                    <line x1={x.f(x.ticks[0])} x2={x.f(r.value)} y1={cy} y2={cy} stroke="var(--line-strong)" />
                    <circle cx={x.f(r.value)} cy={cy} r="5.5" fill={fill} stroke="var(--surface)" strokeWidth="2" />
                  </>
                ) : (
                  <path d={barPath(x.f(0), cy - 8, x.f(r.value), 16)} fill={fill} />
                )}
                <text className={r.highlight ? 'ann' : 'tick'} x={x.f(r.value) + 10} y={cy} dy="0.32em">{fmtInt(r.value)}</text>
              </g>
            );
          })}
        </svg>
        {hover != null && (
          <Tooltip x={x.f(rows[hover].value)} y={m.t + hover * rowH} width={width}>
            <TipRow color={rows[hover].highlight ? color : 'var(--muted)'} label={rows[hover].label} value={`${rows[hover].value.toLocaleString('en-US')} ${spec.unit}`} />
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

export function StoryChart({ chart, eda, compact = false, source }) {
  const inlineImg = typeof window !== 'undefined' ? window.__BEADLINE_INLINE_DATA__?.images?.[chart.src] : null;
  switch (chart.type) {
    case 'line': return <LineChart spec={chart} compact={compact} source={source} />;
    case 'bars': return <RankChart spec={chart} compact={compact} source={source} />;
    case 'dots': return <RankChart spec={chart} dots compact={compact} source={source} />;
    case 'image': return (
      <Figure title={chart.title} source={chart.source || source}>
        <img className="chart-img" src={inlineImg || `${import.meta.env.BASE_URL}data/${chart.src}`} alt={chart.title} />
      </Figure>
    );
    case 'eda:growth': return eda && <GrowthTimeline weekly={eda.weekly} splits={eda.splits} uriStart={eda.uri.start} />;
    case 'eda:hourly': return eda && <HourProfile hourly={eda.hourly} />;
    case 'eda:uri': return eda && <UriPanels uri={eda.uri} />;
    case 'eda:splits': return eda && <SplitRange splits={eda.splits} compact={compact} />;
    default: return null;
  }
}
