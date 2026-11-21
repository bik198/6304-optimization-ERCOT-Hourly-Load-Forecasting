// Chapter "Growth": the weekly timeline with split bands, and average load by year.
import { useMemo, useState } from 'react';
import {
  DataTable, Figure, SPLIT_COLOR, SPLIT_LABEL, Tooltip, TipRow, YGrid, columnPath, fmtDate, fmtGW, fmtInt, fmtK, fmtMW,
  linePath, niceTicks, scale, useWidth,
} from './kit';

const ms = s => Date.parse(`${s.slice(0, 10)}T00:00:00`);

export function GrowthTimeline({ weekly, splits, uriStart }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const H = 320, m = { l: 52, r: 18, t: 40, b: 30 };
  const { x, y, ticks, pts, years } = useMemo(() => {
    const t0 = ms(weekly[0].d), t1 = ms(splits[2].end) + 864e5;
    const vals = weekly.map(w => w.mean);
    const ticks = niceTicks(Math.floor(Math.min(...vals) / 10000) * 10000, Math.max(...vals), 5);
    const x = scale(t0, t1, m.l, width - m.r);
    const y = scale(ticks[0], ticks[ticks.length - 1], H - m.b, m.t);
    const pts = weekly.map(w => [x(ms(w.d)), y(w.mean)]);
    const years = [];
    for (let yr = 2018; yr <= 2026; yr++) years.push(yr);
    return { x, y, ticks, pts, years };
  }, [weekly, splits, width]);

  const onMove = e => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; (pts[mid][0] < px ? (lo = mid) : (hi = mid)); }
    setHover(Math.abs(pts[lo][0] - px) < Math.abs(pts[hi][0] - px) ? lo : hi);
  };
  const uriX = x(ms(uriStart));
  const base = y(ticks[0]);
  const hv = hover != null ? weekly[hover] : null;

  return (
    <Figure
      title="Weekly average ERCOT system load"
      sub="Each point is one week's mean hourly demand. Shaded bands are the chronological model splits."
      table={<DataTable columns={['Week of', 'Mean load (MW)']} rows={weekly.map(w => [w.d, fmtInt(w.mean)])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Weekly average load rose from about 43 GW in 2018 to about 60 GW in 2026, with summer peaks every year.">
          {splits.map(s => {
            const x0 = x(ms(s.start)), x1 = x(ms(s.end) + 864e5);
            return (
              <g key={s.split}>
                <rect x={x0} y={m.t - 26} width={x1 - x0} height={H - m.b - m.t + 26} fill={SPLIT_COLOR[s.split]} opacity="0.07" />
                <rect x={x0 + 8} y={m.t - 18} width="8" height="8" rx="2" fill={SPLIT_COLOR[s.split]} />
                <text className="ann" x={x0 + 22} y={m.t - 10}>{SPLIT_LABEL[s.split]}{x1 - x0 > 150 ? ` · ${s.start.slice(0, 4)}–${s.end.slice(0, 4)}` : ''}</text>
              </g>
            );
          })}
          <YGrid ticks={ticks} y={y} x0={m.l} x1={width - m.r} unit="MW" />
          {years.map(yr => {
            const xx = x(ms(`${yr}-01-01`));
            return <text key={yr} className="tick" x={xx} y={H - m.b + 18} textAnchor="middle">{yr}</text>;
          })}
          <path d={`${linePath(pts)}L${pts[pts.length - 1][0]},${base}L${pts[0][0]},${base}Z`} fill="var(--bright)" fillOpacity="0.06" />
          <path d={linePath(pts)} fill="none" stroke="var(--bright)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
          <g>
            <line x1={uriX} x2={uriX} y1={m.t + 6} y2={base} stroke="var(--brass)" strokeWidth="1.5" />
            <circle cx={uriX} cy={m.t + 6} r="4" fill="var(--brass)" stroke="var(--surface)" strokeWidth="2" />
            <text className="ann" x={uriX + 8} y={m.t + 10}>Winter Storm Uri</text>
          </g>
          {hv && (
            <g pointerEvents="none">
              <line x1={pts[hover][0]} x2={pts[hover][0]} y1={m.t} y2={base} stroke="var(--line-strong)" />
              <circle cx={pts[hover][0]} cy={pts[hover][1]} r="5" fill="var(--bright)" stroke="var(--surface)" strokeWidth="2" />
            </g>
          )}
          <rect x={m.l} y={m.t - 26} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b + 26} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hv && (
          <Tooltip x={pts[hover][0]} y={pts[hover][1]} width={width}>
            <div className="t-head">Week of {fmtDate(hv.d)}</div>
            <div><b>{fmtMW(hv.mean)}</b> average</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

export function YearlyColumns({ yearly }) {
  const [ref, width] = useWidth(400);
  const [hover, setHover] = useState(null);
  const H = 260, m = { l: 44, r: 8, t: 26, b: 28 };
  const max = Math.max(...yearly.map(d => d.mean));
  const ticks = niceTicks(0, max * 1.05, 3);
  const y = scale(0, ticks[ticks.length - 1], H - m.b, m.t);
  const band = (width - m.l - m.r) / yearly.length;
  const bw = Math.min(24, band * 0.55);
  const first = yearly[0], lastFull = [...yearly].reverse().find(d => !d.partial);
  return (
    <Figure
      title="Average load by year"
      sub={`${first.year} → ${lastFull.year}: ${fmtGW(first.mean)} → ${fmtGW(lastFull.mean)} (+${((lastFull.mean / first.mean - 1) * 100).toFixed(0)}%)`}
      table={<DataTable columns={['Year', 'Mean (MW)', 'Peak (MW)', 'Peak hour']} rows={yearly.map(d => [`${d.year}${d.partial ? ' (to Nov 4)' : ''}`, fmtInt(d.mean), fmtInt(d.peak), d.peakAt])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label="Average load by year, 2018 to 2026.">
          <YGrid ticks={ticks} y={y} x0={m.l} x1={width - m.r} unit="MW" />
          {yearly.map((d, i) => {
            const cx = m.l + band * i + band / 2;
            const on = hover === i;
            return (
              <g key={d.year} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x={cx - band / 2} y={m.t} width={band} height={H - m.t - m.b} fill="transparent" />
                <path d={columnPath(cx - bw / 2, y(d.mean), bw, y(0))} fill="var(--bright)" opacity={d.partial ? 0.32 : on ? 1 : 0.86} />
                {(i === 0 || d === lastFull || d.partial || on) && (
                  <text className="ann" x={cx} y={y(d.mean) - 7} textAnchor="middle">{(d.mean / 1000).toFixed(1)}{d.partial ? '*' : ''}</text>
                )}
                <text className="tick" x={cx} y={H - m.b + 18} textAnchor="middle">{`’${String(d.year).slice(2)}`}</text>
              </g>
            );
          })}
        </svg>
        {hover != null && (
          <Tooltip x={m.l + band * hover + band / 2} y={y(yearly[hover].mean)} width={width}>
            <div className="t-head">{yearly[hover].year}{yearly[hover].partial ? ' · Jan 1 – Nov 4' : ''}</div>
            <div><b>{fmtMW(yearly[hover].mean)}</b> average</div>
            <div>Peak {fmtMW(yearly[hover].peak)} · {yearly[hover].peakAt}</div>
          </Tooltip>
        )}
      </div>
      <p className="figure-sub">Labels in GW. *2026 covers Jan 1 – Nov 4.</p>
    </Figure>
  );
}

