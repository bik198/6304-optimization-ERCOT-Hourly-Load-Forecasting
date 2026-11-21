// Chapters "Weather" and "Extremes": the temperature U-curve and the Winter Storm Uri window.
import { useMemo, useState } from 'react';
import {
  DataTable, Figure, Legend, SERIES, Tooltip, TipRow, YGrid, fmtGW, fmtInt, fmtMW, linePath, niceTicks, scale, useWidth,
} from './kit';

export function TemperatureCurve({ temperature }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const H = 340, m = { l: 52, r: 18, t: 26, b: 34 };
  const { x, y, ticks, xticks, minI } = useMemo(() => {
    const ts = temperature.map(d => d.t), vs = temperature.map(d => d.mean);
    const ticks = niceTicks(Math.floor(Math.min(...vs) / 10000) * 10000, Math.max(...vs), 5);
    const x = scale(Math.min(...ts) - 2, Math.max(...ts) + 2, m.l, width - m.r);
    const y = scale(ticks[0], ticks[ticks.length - 1], H - m.b, m.t);
    const xticks = niceTicks(Math.min(...ts), Math.max(...ts), 7).filter(t => t >= Math.min(...ts) - 2 && t <= Math.max(...ts) + 2);
    // the comfort minimum, among bins with enough days to trust
    let minI = 0;
    temperature.forEach((d, i) => { if (d.days >= 20 && d.mean < temperature[minI].mean) minI = i; });
    return { x, y, ticks, xticks, minI };
  }, [temperature, width]);
  const pts = temperature.map(d => [x(d.t), y(d.mean)]);
  const mn = temperature[minI];

  const onMove = e => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(p[0] - px) < Math.abs(pts[best][0] - px)) best = i; });
    setHover(best);
  };

  return (
    <Figure
      title="Daily average load vs daily average temperature"
      sub="Days grouped into 2 °F bins (temperature averaged across 8 weather-zone stations). Open dots: fewer than 20 days in the bin."
      table={<DataTable columns={['Temp bin (°F)', 'Days', 'Mean load (MW)']} rows={temperature.map(d => [`${d.t - 1}–${d.t + 1}`, d.days, fmtInt(d.mean)])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Lowest daily mean load in the ${mn.t} °F bin (${fmtGW(mn.mean)}).`}>
          <YGrid ticks={ticks} y={y} x0={m.l} x1={width - m.r} unit="MW" />
          {xticks.map(t => <text key={t} className="tick" x={x(t)} y={H - m.b + 18} textAnchor="middle">{t}°F</text>)}
          <path d={linePath(pts)} fill="none" stroke="var(--bright)" strokeWidth="2" strokeLinejoin="round" />
          {temperature.map((d, i) => (
            <circle key={d.t} cx={pts[i][0]} cy={pts[i][1]} r="4" stroke="var(--surface)" strokeWidth="2"
              fill={d.days >= 20 ? 'var(--bright)' : 'var(--surface)'} style={d.days >= 20 ? undefined : { stroke: 'var(--bright)', strokeWidth: 1.5 }} />
          ))}
          <g>
            <circle cx={x(mn.t)} cy={y(mn.mean)} r="6" fill="var(--brass)" stroke="var(--surface)" strokeWidth="2" />
            <text className="ann" x={x(mn.t)} y={y(mn.mean) + 22} textAnchor="middle">Lowest: {mn.t} °F bin · {fmtGW(mn.mean)}</text>
          </g>
          {hover != null && <circle cx={pts[hover][0]} cy={pts[hover][1]} r="6" fill="none" stroke="var(--bright)" strokeWidth="2" pointerEvents="none" />}
          <rect x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={H - m.t - m.b} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hover != null && (
          <Tooltip x={pts[hover][0]} y={pts[hover][1]} width={width}>
            <div className="t-head">{temperature[hover].t - 1}–{temperature[hover].t + 1} °F</div>
            <div><b>{fmtMW(temperature[hover].mean)}</b> average</div>
            <div>{temperature[hover].days} day{temperature[hover].days === 1 ? '' : 's'}</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

const hms = s => Date.parse(`${s.replace(' ', 'T')}:00`);

export function UriPanels({ uri }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const m = { l: 52, r: 18 };
  const H1 = 240, H2 = 170, gap = 18, t1 = 22, b2 = 30;
  const hours = uri.hours;
  const { x, yL, yT, lt, tt, days } = useMemo(() => {
    const x = scale(hms(hours[0].t), hms(hours[hours.length - 1].t), m.l, width - m.r);
    const mw = hours.map(h => h.mw), tf = hours.map(h => h.temp).filter(v => v != null);
    const lt = niceTicks(Math.floor(Math.min(...mw) / 10000) * 10000, Math.max(...mw), 5);
    const tt = niceTicks(Math.floor(Math.min(...tf) / 10) * 10, Math.max(...tf), 4);
    const yL = scale(lt[0], lt[lt.length - 1], t1 + H1, t1);
    const top2 = t1 + H1 + gap + 20;
    const yT = scale(tt[0], tt[tt.length - 1], top2 + H2, top2);
    const days = hours.filter(h => h.t.endsWith('00:00')).map(h => h.t.slice(0, 10));
    return { x, yL, yT, lt, tt, days };
  }, [hours, width]);
  const total = t1 + H1 + gap + 20 + H2 + b2;
  const bandX0 = x(hms(`${uri.start} 00:00`)), bandX1 = x(hms(`${uri.end} 23:00`));
  const inUri = hours.filter(h => h.uri);
  const peak = inUri.reduce((a, b) => (b.mw > a.mw ? b : a));
  const coldest = hours.filter(h => h.temp != null).reduce((a, b) => (b.temp < a.temp ? b : a));
  // lowest load in the 48 hours after the storm peak
  const shedEnd = hms(peak.t) + 48 * 36e5;
  const trough = inUri.filter(h => hms(h.t) > hms(peak.t) && hms(h.t) <= shedEnd).reduce((a, b) => (b.mw < a.mw ? b : a));
  const narrow = width < 640;
  const loadPts = hours.map(h => [x(hms(h.t)), yL(h.mw)]);
  const tempPts = hours.filter(h => h.temp != null).map(h => [x(hms(h.t)), yT(h.temp)]);
  const top2 = t1 + H1 + gap + 20;

  const onMove = e => {
    const r = e.currentTarget.getBoundingClientRect();
    const tms = x.invert(e.clientX - r.left);
    const i = Math.round((tms - hms(hours[0].t)) / 36e5);
    setHover(Math.max(0, Math.min(hours.length - 1, i)));
  };
  const hv = hover != null ? hours[hover] : null;

  return (
    <Figure
      title="Winter Storm Uri, February 2021: load and temperature, hour by hour"
      sub="Same time axis, two scales. The brass band is the flagged storm week (Feb 14–20)."
      legend={<Legend items={[{ label: 'System load (MW)', color: 'var(--bright)' }, { label: 'Average temperature (°F)', color: SERIES[0] }]} />}
      table={<DataTable columns={['Hour', 'Load (MW)', 'Temp (°F)', 'Storm flag']} rows={hours.map(h => [h.t, fmtInt(h.mw), h.temp ?? '—', h.uri ? 'yes' : ''])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={total} role="img" aria-label={`Storm peak ${fmtGW(peak.mw)}; lowest temperature ${coldest.temp.toFixed(1)} °F.`}>
          <rect x={bandX0} y={t1 - 8} width={bandX1 - bandX0} height={total - t1 - b2 + 8} fill="var(--brass-soft)" />
          <YGrid ticks={lt} y={yL} x0={m.l} x1={width - m.r} unit="MW" />
          <YGrid ticks={tt} y={yT} x0={m.l} x1={width - m.r} format={v => `${v}°`} />
          {32 >= tt[0] && 32 <= tt[tt.length - 1] && (
            <g>
              <line x1={m.l} x2={width - m.r} y1={yT(32)} y2={yT(32)} stroke={SERIES[0]} strokeOpacity="0.45" strokeWidth="1" />
              <text className="ann-muted" x={width - m.r - 4} y={yT(32) - 6} textAnchor="end">Freezing, 32 °F</text>
            </g>
          )}
          {days.map((d, i) => i % (narrow ? 4 : 2) === 0 && (
            <text key={d} className="tick" x={x(hms(`${d} 00:00`))} y={total - b2 + 18} textAnchor="middle">
              {new Date(`${d}T00:00:00`).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            </text>
          ))}
          <path d={linePath(loadPts)} fill="none" stroke="var(--bright)" strokeWidth="2" strokeLinejoin="round" />
          <path d={linePath(tempPts)} fill="none" stroke={SERIES[0]} strokeWidth="2" strokeLinejoin="round" />
          <g>
            <circle cx={x(hms(peak.t))} cy={yL(peak.mw)} r="5" fill="var(--brass)" stroke="var(--surface)" strokeWidth="2" />
            <text className="ann" x={x(hms(peak.t)) - 10} y={yL(peak.mw) - 4} textAnchor="end">{narrow ? fmtGW(peak.mw) : `Storm peak ${fmtGW(peak.mw)}`}</text>
            <circle cx={x(hms(trough.t))} cy={yL(trough.mw)} r="5" fill="var(--bright)" stroke="var(--surface)" strokeWidth="2" />
            <text className="ann" x={x(hms(trough.t)) + 8} y={yL(trough.mw) + 22}>{fmtGW(trough.mw)}</text>
            <circle cx={x(hms(coldest.t))} cy={yT(coldest.temp)} r="5" fill={SERIES[0]} stroke="var(--surface)" strokeWidth="2" />
            <text className="ann" x={x(hms(coldest.t)) + 10} y={yT(coldest.temp) + 4}>{coldest.temp.toFixed(1)} °F</text>
          </g>
          <text className="label" x={m.l} y={top2 - 12}>Temperature</text>
          {hv && (
            <g pointerEvents="none">
              <line x1={x(hms(hv.t))} x2={x(hms(hv.t))} y1={t1} y2={total - b2} stroke="var(--line-strong)" />
              <circle cx={x(hms(hv.t))} cy={yL(hv.mw)} r="4.5" fill="var(--bright)" stroke="var(--surface)" strokeWidth="2" />
              {hv.temp != null && <circle cx={x(hms(hv.t))} cy={yT(hv.temp)} r="4.5" fill={SERIES[0]} stroke="var(--surface)" strokeWidth="2" />}
            </g>
          )}
          <rect x={m.l} y={t1} width={Math.max(0, width - m.l - m.r)} height={total - t1 - b2} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hv && (
          <Tooltip x={x(hms(hv.t))} y={yL(hv.mw)} width={width}>
            <div className="t-head">{new Date(hv.t.replace(' ', 'T')).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' })}{hv.uri ? ' · storm' : ''}</div>
            <TipRow color="var(--bright)" label="Load" value={fmtMW(hv.mw)} />
            {hv.temp != null && <TipRow color={SERIES[0]} label="Temperature" value={`${hv.temp.toFixed(1)} °F`} />}
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}
