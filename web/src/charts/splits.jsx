// Chronological train / validation / test split: exact-100% share bar + load range per split.
import { useState } from 'react';
import { DataTable, Figure, Legend, SPLIT_COLOR, SPLIT_LABEL, Tooltip, fmtGW, fmtInt, fmtK, fmtMW, niceTicks, scale, useWidth } from './kit';

/** Largest-remainder integer percentages so shares always sum to exact 100%. */
function exactPercents(splits) {
  const total = splits.reduce((a, s) => a + s.hours, 0) || 1;
  const raw = splits.map(s => (s.hours / total) * 100);
  const floors = raw.map(v => Math.floor(v));
  let rem = 100 - floors.reduce((a, b) => a + b, 0);
  const order = raw
    .map((v, i) => ({ i, frac: v - floors[i] }))
    .sort((a, b) => b.frac - a.frac);
  for (let k = 0; k < rem; k++) floors[order[k].i] += 1;
  return floors;
}

/** Segmented bar: the 77,496 hours in chronological order, embedded directly inside SplitRange. */
export function SplitBar({ splits }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const total = splits.reduce((a, s) => a + s.hours, 0);
  const pcts = exactPercents(splits);
  const H = 26;
  let cx = 0;
  return (
    <div className="chart split-bar-inline" ref={ref}>
      <svg width={width} height={H} role="img" aria-label={splits.map((s, i) => `${SPLIT_LABEL[s.split]} ${pcts[i]}%`).join(', ')}>
        {splits.map((s, i) => {
          const w = (s.hours / total) * width;
          const x0 = cx;
          cx += w;
          const label = `${SPLIT_LABEL[s.split]} · ${pcts[i]}%`;
          const short = `${pcts[i]}%`;
          const text = w > label.length * 7.2 + 16 ? label : w > short.length * 7.2 + 12 ? short : '';
          return (
            <g key={s.split} onPointerEnter={() => setHover({ i, x: x0 + w / 2 })} onPointerLeave={() => setHover(null)}>
              <rect x={x0 + (i ? 1.5 : 0)} y="0" width={Math.max(0, w - (i ? 2.5 : 1))} height={H} rx="4" fill={SPLIT_COLOR[s.split]} />
              <text x={x0 + 10} y={H / 2} dy="0.35em" fontSize="11.5" fontWeight="650" fill="#fff" pointerEvents="none">{text}</text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <Tooltip x={hover.x} y={0} width={width}>
          <div className="t-head">{SPLIT_LABEL[splits[hover.i].split]} ({pcts[hover.i]}%)</div>
          <div><b>{fmtInt(splits[hover.i].hours)}</b> hours · {splits[hover.i].start} → {splits[hover.i].end}</div>
        </Tooltip>
      )}
    </div>
  );
}

/** Range per split (min → peak) with the mean, against the 2018–2023 training maximum. */
export function SplitRange({ splits, compact = false }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const train = splits[0];
  const m = { l: 92, r: 124, t: 28, b: 28 };
  const rowH = compact ? 42 : 48;
  const H = m.t + splits.length * rowH + m.b;
  const ticks = niceTicks(Math.floor(Math.min(...splits.map(s => s.min)) / 10000) * 10000, Math.max(...splits.map(s => s.peak)), 5);
  const x = scale(ticks[0], ticks[ticks.length - 1], m.l, width - m.r);
  return (
    <Figure
      title="Chronological split shares & load range"
      sub={`Load in MW: minimum to maximum per split, dot at the mean, and vertical reference line at the training maximum (${fmtGW(train.peak)}).`}
      legend={<Legend items={splits.map(s => ({ label: SPLIT_LABEL[s.split], color: SPLIT_COLOR[s.split] }))} />}
      table={<DataTable columns={['Split', 'Period', 'Hours', 'Days (samples)', 'Min (MW)', 'Mean (MW)', 'Peak (MW)']} rows={splits.map(s => [SPLIT_LABEL[s.split], `${s.start} → ${s.end}`, fmtInt(s.hours), s.days != null ? fmtInt(s.days) : '—', fmtInt(s.min), fmtInt(s.mean), fmtInt(s.peak)])} />}
    >
      <SplitBar splits={splits} />
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`The test split peaks at ${fmtGW(splits[2].peak)}, ${fmtGW(splits[2].peak - train.peak)} above the training maximum.`}>
          {ticks.map(t => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.t - 6} y2={H - m.b} stroke="var(--grid)" />
              <text className="tick" x={x(t)} y={H - m.b + 18} textAnchor="middle">{fmtK(t)}</text>
            </g>
          ))}
          <line x1={x(train.peak)} x2={x(train.peak)} y1={m.t - 10} y2={H - m.b} stroke="var(--brass)" strokeWidth="1.5" strokeDasharray="4 3" />
          <text className="ann" x={x(train.peak) - 6} y={m.t - 13} textAnchor="end">Training max {fmtGW(train.peak)}</text>
          {splits.map((s, i) => {
            const yy = m.t + i * rowH + rowH / 2;
            const over = s.peak > train.peak;
            return (
              <g key={s.split} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x="0" y={yy - rowH / 2} width={width} height={rowH} fill="transparent" />
                <text className="tick" x={m.l - 12} y={yy} dy="0.32em" textAnchor="end" style={{ fill: 'var(--bright)', fontWeight: 600 }}>{SPLIT_LABEL[s.split]}</text>
                <line x1={x(s.min)} x2={x(s.peak)} y1={yy} y2={yy} stroke={SPLIT_COLOR[s.split]} strokeWidth="4.5" strokeLinecap="round" />
                <circle cx={x(s.mean)} cy={yy} r="6" fill={SPLIT_COLOR[s.split]} stroke="var(--surface)" strokeWidth="2" />
                <text className="ann" x={x(s.peak) + 8} y={yy} dy="0.32em" textAnchor="start">
                  {fmtGW(s.peak)}{over ? ` (+${((s.peak - train.peak) / 1000).toFixed(1)})` : ''}
                </text>
              </g>
            );
          })}
        </svg>
        {hover != null && (
          <Tooltip x={x(splits[hover].mean)} y={m.t + hover * rowH + 6} width={width}>
            <div className="t-head">{SPLIT_LABEL[splits[hover].split]} · {splits[hover].start} → {splits[hover].end}</div>
            <div>Min {fmtMW(splits[hover].min)}</div>
            <div>Mean <b>{fmtMW(splits[hover].mean)}</b></div>
            <div>Peak {fmtMW(splits[hover].peak)}</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}
