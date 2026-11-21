// Chapters "The day" and "The year": hour-of-day profiles and the month × hour heatmap.
import { useMemo, useState } from 'react';
import {
  DataTable, Figure, Legend, MONTHS, SEASON_COLOR, SERIES, Tooltip, TipRow, YGrid, fmtGW, fmtInt, fmtMW, hourLabel,
  linePath, niceTicks, scale, useTheme, useWidth,
} from './kit';

const SEASONS = [
  { key: 'winter', label: 'Winter (Dec–Feb)' },
  { key: 'spring', label: 'Spring (Mar–May)' },
  { key: 'summer', label: 'Summer (Jun–Aug)' },
  { key: 'fall', label: 'Fall (Sep–Nov)' },
];

export function HourProfile({ hourly }) {
  const [mode, setMode] = useState('seasons');
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const H = 340, m = { l: 52, r: 18, t: 22, b: 30 };

  const series = mode === 'seasons'
    ? SEASONS.map(s => ({ key: s.key, label: s.label, color: SEASON_COLOR[s.key], values: hourly.seasons[s.key] }))
    : [
      { key: 'weekday', label: 'Weekdays', color: SERIES[0], values: hourly.weekday },
      { key: 'weekend', label: 'Weekends', color: SERIES[1], values: hourly.weekend },
    ];

  const { x, y, ticks } = useMemo(() => {
    const all = series.flatMap(s => s.values);
    const ticks = niceTicks(Math.floor(Math.min(...all) / 10000) * 10000, Math.max(...all), 5);
    return { x: scale(0, 23, m.l, width - m.r), y: scale(ticks[0], ticks[ticks.length - 1], H - m.b, m.t), ticks };
  }, [series, width]);

  const summer = hourly.seasons.summer;
  const sPeakH = summer.indexOf(Math.max(...summer));
  const winter = hourly.seasons.winter;
  const wMornH = winter.slice(5, 11).indexOf(Math.max(...winter.slice(5, 11))) + 5;

  const onMove = e => {
    const r = e.currentTarget.getBoundingClientRect();
    setHover(Math.max(0, Math.min(23, Math.round(x.invert(e.clientX - r.left)))));
  };

  return (
    <Figure
      title="Average load by hour of day"
      sub={mode === 'seasons' ? 'Mean demand for each clock hour, by season' : 'Mean demand for each clock hour, Monday–Friday vs Saturday–Sunday'}
      controls={
        <div className="seg" role="group" aria-label="Compare by">
          <button type="button" aria-pressed={mode === 'seasons'} onClick={() => setMode('seasons')}>Seasons</button>
          <button type="button" aria-pressed={mode === 'week'} onClick={() => setMode('week')}>Weekday vs weekend</button>
        </div>
      }
      legend={<Legend items={series} />}
      table={<DataTable columns={['Hour', ...series.map(s => s.label)]} rows={Array.from({ length: 24 }, (_, h) => [hourLabel(h), ...series.map(s => fmtInt(s.values[h]))])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Summer mean peaks at ${hourLabel(sPeakH)} (${fmtGW(summer[sPeakH])}).`}>
          <YGrid ticks={ticks} y={y} x0={m.l} x1={width - m.r} unit="MW" />
          {[0, 3, 6, 9, 12, 15, 18, 21, 23].map(h => <text key={h} className="tick" x={x(h)} y={H - m.b + 18} textAnchor="middle">{hourLabel(h)}</text>)}
          {series.map(s => (
            <path key={`${mode}-${s.key}`} d={linePath(s.values.map((v, h) => [x(h), y(v)]))} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {mode === 'seasons' && (
            <g>
              <circle cx={x(sPeakH)} cy={y(summer[sPeakH])} r="4.5" fill={SEASON_COLOR.summer} stroke="var(--surface)" strokeWidth="2" />
              <text className="ann" x={x(sPeakH) - 10} y={y(summer[sPeakH]) - 10} textAnchor="end">Summer {hourLabel(sPeakH)} · {fmtGW(summer[sPeakH])}</text>
              <circle cx={x(wMornH)} cy={y(winter[wMornH])} r="4.5" fill={SEASON_COLOR.winter} stroke="var(--surface)" strokeWidth="2" />
              <text className="ann" x={x(wMornH)} y={y(winter[wMornH]) - 12} textAnchor="middle">Winter {hourLabel(wMornH)} · {fmtGW(winter[wMornH])}</text>
            </g>
          )}
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={m.t} y2={H - m.b} stroke="var(--line-strong)" pointerEvents="none" />}
          {hover != null && series.map(s => <circle key={s.key} cx={x(hover)} cy={y(s.values[hover])} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" pointerEvents="none" />)}
          <rect x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hover != null && (
          <Tooltip x={x(hover)} y={y(Math.max(...series.map(s => s.values[hover])))} width={width}>
            <div className="t-head">{hourLabel(hover)}</div>
            {[...series].sort((a, b) => b.values[hover] - a.values[hover]).map(s => <TipRow key={s.key} color={s.color} label={s.label.split(' (')[0]} value={fmtMW(s.values[hover])} />)}
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

// sequential blue ramp (palette steps 100 → 700). The lowest load takes the step nearest the
// surface: the lightest step on light themes, the darkest step on the dark theme.
const RAMP_LIGHT = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const RAMP_DARK = [...RAMP_LIGHT].reverse();
const rampColor = (RAMP, k) => {
  const p = Math.max(0, Math.min(1, k)) * (RAMP.length - 1);
  const i = Math.floor(p), f = p - i;
  if (i >= RAMP.length - 1) return RAMP[RAMP.length - 1];
  const a = parseInt(RAMP[i].slice(1), 16), b = parseInt(RAMP[i + 1].slice(1), 16);
  const ch = s => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
};

export function MonthHourHeatmap({ heatmap }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const dark = useTheme() === 'dark';
  const RAMP = dark ? RAMP_DARK : RAMP_LIGHT;
  const m = { l: 40, r: 6, t: 22, b: 8 };
  const cw = (width - m.l - m.r) / 24;
  const ch = Math.max(16, Math.min(30, cw * 0.9));
  const H = m.t + ch * 12 + m.b;
  const flat = heatmap.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  let peak = { m: 0, h: 0 };
  heatmap.forEach((row, mi) => row.forEach((v, h) => { if (v > heatmap[peak.m][peak.h]) peak = { m: mi, h }; }));

  return (
    <Figure
      title="Average load by month and hour"
      sub={`${dark ? 'Lighter' : 'Darker'} means more load. Each cell is the mean of every hour with that month and clock hour.`}
      legend={
        <div className="legend" aria-hidden="true">
          <span>{fmtGW(lo)}</span>
          <span style={{ flex: '0 1 220px', height: 10, borderRadius: 5, background: `linear-gradient(90deg, ${RAMP.join(',')})` }} />
          <span>{fmtGW(hi)}</span>
        </div>
      }
      table={<DataTable columns={['Month', ...Array.from({ length: 24 }, (_, h) => String(h).padStart(2, '0'))]} rows={heatmap.map((row, mi) => [MONTHS[mi], ...row.map(v => fmtInt(v))])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Highest mean load: ${MONTHS[peak.m]} at ${hourLabel(peak.h)}.`}>
          {Array.from({ length: 24 }, (_, h) => h % 3 === 0 && <text key={h} className="tick" x={m.l + cw * h + cw / 2} y={m.t - 8} textAnchor="middle">{String(h).padStart(2, '0')}</text>)}
          {heatmap.map((row, mi) => (
            <g key={mi}>
              <text className="tick" x={m.l - 8} y={m.t + ch * mi + ch / 2} dy="0.32em" textAnchor="end">{MONTHS[mi]}</text>
              {row.map((v, h) => (
                <rect key={h}
                  x={m.l + cw * h + 1} y={m.t + ch * mi + 1} width={Math.max(0, cw - 2)} height={ch - 2} rx="3"
                  fill={rampColor(RAMP, (v - lo) / (hi - lo))}
                  stroke={hover && hover.m === mi && hover.h === h ? 'var(--bright)' : 'none'} strokeWidth="2"
                  onPointerEnter={() => setHover({ m: mi, h })} onPointerLeave={() => setHover(null)} />
              ))}
            </g>
          ))}
        </svg>
        {hover && (
          <Tooltip x={m.l + cw * hover.h + cw / 2} y={m.t + ch * hover.m} width={width}>
            <div className="t-head">{MONTHS[hover.m]} · {hourLabel(hover.h)}</div>
            <div><b>{fmtMW(heatmap[hover.m][hover.h])}</b> average</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}
