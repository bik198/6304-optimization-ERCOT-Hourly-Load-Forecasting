// Chapter "The tail": how often each load level happens, and where the extremes sit.
import { useState } from 'react';
import { DataTable, Figure, Tooltip, YGrid, columnPath, fmtGW, fmtInt, fmtK, niceTicks, scale, useWidth } from './kit';

export function LoadHistogram({ distribution }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const { bins, p50, p99, max } = distribution;
  const total = bins.reduce((a, b) => a + b.hours, 0);
  const H = 300, m = { l: 52, r: 18, t: 34, b: 30 };
  const x = scale(bins[0].lo, bins[bins.length - 1].hi, m.l, width - m.r);
  const ticks = niceTicks(0, Math.max(...bins.map(b => b.hours)), 4);
  const y = scale(0, ticks[ticks.length - 1], H - m.b, m.t);
  const xt = niceTicks(bins[0].lo, bins[bins.length - 1].hi, 6).filter(t => t >= bins[0].lo && t <= bins[bins.length - 1].hi);
  const marks = [
    { v: p50, label: `Median ${fmtGW(p50)}`, color: 'var(--muted)' },
    { v: p99, label: `99th pct ${fmtGW(p99)}`, color: 'var(--bright)' },
    { v: max, label: `Record ${fmtGW(max)}`, color: 'var(--brass)' },
  ];
  return (
    <Figure
      title="How many hours sit at each load level"
      sub="Hourly system load, 2 GW bins. Brass bars: bins at or above the 99th percentile."
      table={<DataTable columns={['Load bin (MW)', 'Hours', 'Share']} rows={bins.map(b => [`${fmtInt(b.lo)}–${fmtInt(b.hi)}`, fmtInt(b.hours), `${((b.hours / total) * 100).toFixed(2)}%`])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Median ${fmtGW(p50)}, 99th percentile ${fmtGW(p99)}, maximum ${fmtGW(max)}.`}>
          <YGrid ticks={ticks} y={y} x0={m.l} x1={width - m.r} format={v => fmtK(v)} unit="Hours" />
          {xt.map(t => <text key={t} className="tick" x={x(t)} y={H - m.b + 18} textAnchor="middle">{fmtK(t)}</text>)}
          {bins.map((b, i) => {
            const x0 = x(b.lo) + 1, w = Math.max(0, x(b.hi) - x(b.lo) - 2);
            return (
              <g key={b.lo} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x={x(b.lo)} y={m.t} width={x(b.hi) - x(b.lo)} height={H - m.t - m.b} fill="transparent" />
                <path d={columnPath(x0, y(b.hours), w, y(0))} fill={b.lo >= p99 ? 'var(--brass)' : 'var(--bright)'} opacity={hover === i ? 1 : b.lo >= p99 ? 0.95 : 0.78} />
              </g>
            );
          })}
          {marks.map((mk, i) => (
            <g key={mk.label} pointerEvents="none">
              <line x1={x(mk.v)} x2={x(mk.v)} y1={m.t - 4} y2={H - m.b} stroke={mk.color} strokeWidth="1.5" />
              <text className="ann" x={x(mk.v) + (i === 0 ? 6 : -6)} y={m.t - 10 + (i === 1 ? 16 : 0)} textAnchor={i === 0 ? 'start' : 'end'}>{mk.label}</text>
            </g>
          ))}
        </svg>
        {hover != null && (
          <Tooltip x={x((bins[hover].lo + bins[hover].hi) / 2)} y={y(bins[hover].hours)} width={width}>
            <div className="t-head">{fmtK(bins[hover].lo)}–{fmtK(bins[hover].hi)} MW</div>
            <div><b>{fmtInt(bins[hover].hours)}</b> hours · {((bins[hover].hours / total) * 100).toFixed(1)}%</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}
