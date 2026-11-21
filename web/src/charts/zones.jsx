// Chapter "Where": zone growth (bars) and zone share of the system (tall segmented bars).
import { useState } from 'react';
import { DataTable, Figure, Legend, SERIES, Tooltip, barPath, fmtInt, fmtMW, fmtPct, useWidth } from './kit';

export function ZoneGrowth({ zones }) {
  const [ref, width] = useWidth(500);
  const [hover, setHover] = useState(null);
  const rows = [...zones.growth].sort((a, b) => b.growthPct - a.growthPct);
  const top = rows[0];
  const m = { l: 108, r: 64, t: 6 };
  const rowH = 34, barH = 18;
  const H = m.t + rows.length * rowH + 4;
  const max = Math.max(...rows.map(r => r.growthPct));
  const x = v => m.l + (v / max) * (width - m.l - m.r);
  return (
    <Figure
      title={`Growth in average load, ${zones.years[0]} → ${zones.years[zones.years.length - 1]}`}
      sub="Percent change in each weather zone's mean hourly load"
      table={<DataTable columns={['Zone', `${zones.years[0]} mean (MW)`, `${zones.years[zones.years.length - 1]} mean (MW)`, 'Change']} rows={rows.map(r => [r.label, fmtInt(r.mw2018), fmtInt(r.mw2025), fmtPct(r.growthPct)])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`${top.label} grew fastest, ${fmtPct(top.growthPct, 0)}.`}>
          <line x1={m.l} x2={m.l} y1={m.t} y2={H - 4} stroke="var(--axis)" />
          {rows.map((r, i) => {
            const yy = m.t + i * rowH + (rowH - barH) / 2;
            const lead = i === 0;
            return (
              <g key={r.zone} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={m.t + i * rowH} width={width} height={rowH} fill="transparent" />
                <text className="tick" x={m.l - 10} y={yy + barH / 2} dy="0.32em" textAnchor="end" style={{ fill: lead ? 'var(--bright)' : undefined, fontWeight: lead ? 650 : 400 }}>{r.label}</text>
                <path d={barPath(m.l, yy, x(r.growthPct), barH)} fill={lead ? 'var(--brass)' : 'var(--bright)'} opacity={lead || hover === i ? 1 : 0.8} />
                <text className="ann" x={x(r.growthPct) + 8} y={yy + barH / 2} dy="0.32em">{fmtPct(r.growthPct, 0)}</text>
              </g>
            );
          })}
        </svg>
        {hover != null && (
          <Tooltip x={x(rows[hover].growthPct)} y={m.t + hover * rowH} width={width}>
            <div className="t-head">{rows[hover].label}</div>
            <div>{fmtMW(rows[hover].mw2018)} → <b>{fmtMW(rows[hover].mw2025)}</b></div>
            <div>{fmtPct(rows[hover].growthPct)}</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}

export function ZoneShare({ zones }) {
  const [ref, width] = useWidth(500);
  const [hover, setHover] = useState(null);
  const years = [zones.years[0], zones.years[zones.years.length - 1]];
  const idx = [0, zones.years.length - 1];
  const items = zones.order.map((z, i) => ({ zone: z, label: zones.labels[z], color: SERIES[i] }));
  const m = { l: 44, r: 4 };
  const barH = 28, rowGap = 26;
  const H = 2 * barH + rowGap + 6;
  const inner = Math.max(0, width - m.l - m.r);

  const rows = idx.map(k => {
    const vals = zones.order.map(z => zones.meanByYear[z][k]);
    const total = vals.reduce((a, b) => a + b, 0);
    return { total, parts: vals.map((v, i) => ({ ...items[i], v, share: v / total })) };
  });

  return (
    <Figure
      title="Each zone's share of system load"
      sub={`${years[0]} vs ${years[1]}, share of the mean hourly total`}
      legend={<Legend items={items} box />}
      table={<DataTable columns={['Zone', `${years[0]} share`, `${years[1]} share`]} rows={items.map((it, i) => [it.label, `${(rows[0].parts[i].share * 100).toFixed(1)}%`, `${(rows[1].parts[i].share * 100).toFixed(1)}%`])} />}
    >
      <div className="chart" ref={ref}>
        <svg width={width} height={H} role="img" aria-label={`Far West share: ${(rows[0].parts[2].share * 100).toFixed(1)}% in ${years[0]}, ${(rows[1].parts[2].share * 100).toFixed(1)}% in ${years[1]}.`}>
          {rows.map((row, ri) => {
            let cx = m.l;
            const yy = ri * (barH + rowGap);
            return (
              <g key={ri}>
                <text className="tick" x={m.l - 8} y={yy + barH / 2} dy="0.32em" textAnchor="end">{years[ri]}</text>
                {row.parts.map((p, i) => {
                  const w = p.share * inner;
                  const x0 = cx;
                  cx += w;
                  const label = `${(p.share * 100).toFixed(0)}%`;
                  const fits = w - 2 > label.length * 7 + 10;
                  const dark = p.color === SERIES[3];
                  return (
                    <g key={p.zone} onPointerEnter={() => setHover({ ri, i, x: x0 + w / 2, y: yy })} onPointerLeave={() => setHover(null)}>
                      <rect x={x0 + (i ? 1 : 0)} y={yy} width={Math.max(0, w - (i ? 2 : 1))} height={barH} rx="4" fill={p.color}
                        stroke={hover && hover.ri === ri && hover.i === i ? 'var(--bright)' : 'none'} strokeWidth="2" />
                      {fits && <text x={x0 + 8} y={yy + barH / 2} dy="0.34em" fontSize="11.5" fontWeight="650" fill={dark ? '#0b0b0b' : '#fff'} pointerEvents="none">{label}</text>}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {hover && (
          <Tooltip x={hover.x} y={hover.y} width={width}>
            <div className="t-head">{rows[hover.ri].parts[hover.i].label} · {years[hover.ri]}</div>
            <div><b>{(rows[hover.ri].parts[hover.i].share * 100).toFixed(1)}%</b> of system load</div>
            <div>{fmtMW(rows[hover.ri].parts[hover.i].v)} average</div>
          </Tooltip>
        )}
      </div>
    </Figure>
  );
}
