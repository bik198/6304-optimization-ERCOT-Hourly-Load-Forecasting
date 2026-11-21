// GridStory: the ERCOT project as a single scrolling page on a flat white grid.
// Step 0 (EDA) chapters are live; later steps are listed until their data exists.
import { useEffect, useState } from 'react';
import Chapter from './story/Chapter';
import ThemeSwitch from './story/ThemeSwitch';
import { Figure, SPLIT_COLOR, SPLIT_LABEL, fmtGW, fmtInt, hourLabel } from './charts/kit';
import { GrowthTimeline, YearlyColumns } from './charts/growth';
import { HourProfile, MonthHourHeatmap } from './charts/daily';
import { TemperatureCurve, UriPanels } from './charts/weather';
import { ZoneGrowth, ZoneShare } from './charts/zones';
import { LoadHistogram } from './charts/tail';
import { SplitBar, SplitRange } from './charts/splits';

const CHAPTERS = [
  ['data', 'Data'], ['growth', 'Growth'], ['day', 'Day'], ['year', 'Year'], ['weather', 'Weather'],
  ['zones', 'Zones'], ['uri', 'Uri'], ['tail', 'Tail'], ['splits', 'Splits'], ['next', 'Next'],
];

const NEXT_STEPS = [
  ['Step 1', 'Baselines', 'Previous-day and previous-week seasonal-naive forecasts; AR(p) with AIC order selection, ACF/PACF and Ljung-Box checks.'],
  ['Step 2', 'Ridge & SVR', 'Multi-output Ridge and RBF-SVR; gradient descent vs conjugate gradient on the Ridge objective.'],
  ['Step 3', 'LSTM & optimizers', 'PyTorch LSTM; full-batch gradient descent with backtracking vs mini-batch SGD on a fixed feed-forward network.'],
  ['Step 4', 'Battery dispatch', 'Convex QP (CVXPY) over hourly charge, discharge and state of charge: daily peak plus wear penalty.'],
  ['Step 5', 'Regret', 'Each model’s test forecasts through the same QP: peak reduction, throughput, feasibility, regret vs perfect foresight.'],
  ['Step 6', 'Motion', '60 fps motion-graphics video rendered from the JSON outputs of Steps 1–5.'],
];

const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const toMs = t => Date.parse(t.replace(' ', 'T'));
const fmtHour = t => new Date(t.replace(' ', 'T')).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' });

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/eda.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setError(e.message));
  }, []);

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#top">GridStory<b>.</b></a>
        <nav className="nav" aria-label="Chapters">
          {CHAPTERS.map(([id, label]) => <a key={id} href={`#${id}`}>{label}</a>)}
        </nav>
        <ThemeSwitch />
        <span className="label">Step 0 · EDA</span>
      </header>

      {!data ? (
        <div className="loading">{error ? `Could not load data/eda.json (${error}). Run: python export_web_data.py` : 'Loading…'}</div>
      ) : (
        <Story data={data} />
      )}

      <footer className="footer">
        <span>GridStory · ASDS 6304 · Group 3</span>
        <span>Data: ERCOT hourly load & zone weather, 2018-01-01 → 2026-11-04</span>
      </footer>
    </>
  );
}

function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
      {note && <span className="note">{note}</span>}
    </div>
  );
}

function Story({ data }) {
  const { meta, yearly, hourly, splits, distribution, zones, uri } = data;
  const overall = hourly.overall;
  const troughH = overall.indexOf(Math.min(...overall));
  const peakH = overall.indexOf(Math.max(...overall));
  const first = yearly[0];
  const lastFull = [...yearly].reverse().find(y => !y.partial);
  const ytd = yearly.find(y => y.partial);
  const record = yearly.reduce((a, b) => (b.peak > a.peak ? b : a));
  const summerPeaks = yearly.filter(y => /-0[78]-/.test(y.peakAt) && /(16|17):00$/.test(y.peakAt)).length;
  const train = splits[0], test = splits[2];
  const total = splits.reduce((a, s) => a + s.hours, 0);
  const farWest = zones.growth.find(z => z.zone === 'farWest');
  const fwShare = k => (zones.meanByYear.farWest[k] / zones.order.reduce((a, z) => a + zones.meanByYear[z][k], 0)) * 100;
  const winter = hourly.seasons.winter;
  const wMornH = winter.slice(5, 11).indexOf(Math.max(...winter.slice(5, 11))) + 5;
  const summer = hourly.seasons.summer;
  const sPeakH = summer.indexOf(Math.max(...summer));
  let hm = { m: 0, h: 0 };
  data.heatmap.forEach((row, mi) => row.forEach((v, h) => { if (v > data.heatmap[hm.m][hm.h]) hm = { m: mi, h }; }));
  const temps = data.temperature;
  const tMin = temps.filter(d => d.days >= 20).reduce((a, b) => (b.mean < a.mean ? b : a));
  const tCold = temps[0], tHot = temps[temps.length - 1];
  const uriHours = uri.hours.filter(h => h.uri);
  const uriPeak = uriHours.reduce((a, b) => (b.mw > a.mw ? b : a));
  const uriAfter = uriHours.filter(h => toMs(h.t) > toMs(uriPeak.t) && toMs(h.t) <= toMs(uriPeak.t) + 48 * 36e5)
    .reduce((a, b) => (b.mw < a.mw ? b : a));
  const uriColdest = uri.hours.filter(h => h.temp != null).reduce((a, b) => (b.temp < a.temp ? b : a));

  return (
    <main id="top" className="story">
      <section className="intro">
        <h1>GridStory<span className="dot">.</span></h1>
        <p>ERCOT hourly system load, {meta.start} → {meta.end}. {fmtInt(meta.hours)} hours, {meta.zones} weather zones.</p>
        <p>Project goal: forecast the next day's 24 hourly loads, then schedule a battery to reduce the daily peak.</p>
      </section>

      <Chapter id="data" no="01" eyebrow="The data" title="619,968 rows → 77,496 hours">
        <p>
          The raw table has <strong>{fmtInt(meta.rawRows)}</strong> rows: {fmtInt(meta.hours)} hours × {meta.zones} ERCOT weather zones.
          The system total is identical across all {meta.zones} zone rows (maximum difference: {meta.maxZoneSpread.toFixed(3)} MW). Step 0 keeps one row per hour
          and builds {fmtInt(meta.samples)} day-ahead samples.
        </p>
        <div className="stats">
          <Stat label="Raw rows" value={fmtInt(meta.rawRows)} note={`${fmtInt(meta.hours)} hours × ${meta.zones} zones`} />
          <Stat label="Unique hours" value={fmtInt(meta.hours)} note={`${meta.start} → ${meta.end}`} />
          <Stat label="Day-ahead samples" value={fmtInt(meta.samples)} note={`${meta.features} features → 24 hourly targets`} />
          <Stat label="Storm hours flagged" value={meta.uriHours} note="Winter Storm Uri, Feb 14–20, 2021" />
        </div>
        <ul className="facts">
          <li><strong>{meta.dstRepeats}</strong> repeated 01:00 hours (daylight-saving fall-back)</li>
          <li><strong>{meta.missingDates.length}</strong> missing day ({meta.missingDates.join(', ')}), handled in lag construction</li>
          <li><strong>67</strong> hours without temperature, all in train, filled with train medians</li>
        </ul>
      </Chapter>

      <Chapter id="growth" no="02" eyebrow="Growth" title={`Average load: ${fmtGW(first.mean)} → ${fmtGW(lastFull.mean)}`}>
        <p>
          Mean hourly load was <strong>{fmtGW(first.mean)}</strong> in {first.year} and <strong>{fmtGW(lastFull.mean)}</strong> in {lastFull.year},
          a change of +{((lastFull.mean / first.mean - 1) * 100).toFixed(0)}%. The {ytd.year} mean (Jan 1 – Nov 4) is {fmtGW(ytd.mean)}.
        </p>
        <GrowthTimeline weekly={data.weekly} splits={splits} uriStart={uri.start} />
        <div className="stats">
          <Stat label="Annual peaks in Jul/Aug, 16:00–17:00" value={`${summerPeaks} of ${yearly.length}`} note={`${first.year}–${ytd.year}`} />
          <Stat label="All-time maximum" value={fmtGW(record.peak)} note={`${record.peakAt} (test split)`} />
          <Stat label="Mean growth per year" value={`+${(((lastFull.mean / first.mean) ** (1 / (lastFull.year - first.year)) - 1) * 100).toFixed(1)}%`} note={`Compound, ${first.year} → ${lastFull.year}`} />
        </div>
        <YearlyColumns yearly={yearly} />
      </Chapter>

      <Chapter id="day" no="03" eyebrow="The day" title={`Lowest at ${hourLabel(troughH)}, highest at ${hourLabel(peakH)}`}>
        <p>
          Mean load by clock hour is lowest at <strong>{hourLabel(troughH)}</strong> ({fmtGW(overall[troughH])}) and highest at <strong>{hourLabel(peakH)}</strong> ({fmtGW(overall[peakH])}),
          a difference of {((overall[peakH] / overall[troughH] - 1) * 100).toFixed(0)}%. The summer mean peaks at {hourLabel(sPeakH)} ({fmtGW(summer[sPeakH])}).
          The winter mean has a local maximum at {hourLabel(wMornH)} ({fmtGW(winter[wMornH])}).
        </p>
        <HourProfile hourly={hourly} />
      </Chapter>

      <Chapter id="year" no="04" eyebrow="The year" title={`Highest cell: ${MONTH_NAMES[hm.m]}, ${hourLabel(hm.h)}`}>
        <p>
          Mean load for each month × clock hour. The highest cell is <strong>{MONTH_NAMES[hm.m]} at {hourLabel(hm.h)}</strong> ({fmtGW(data.heatmap[hm.m][hm.h])}).
          The day-ahead feature set includes deterministic calendar features.
        </p>
        <MonthHourHeatmap heatmap={data.heatmap} />
      </Chapter>

      <Chapter id="weather" no="05" eyebrow="Weather" title="Load vs temperature">
        <p>
          Daily mean load is lowest in the <strong>{tMin.t} °F</strong> bin ({fmtGW(tMin.mean)}). It is {fmtGW(tHot.mean)} in the {tHot.t} °F bin and {fmtGW(tCold.mean)} in the {tCold.t} °F bin.
          Observed target-day weather is excluded from the primary forecasting inputs (look-ahead leakage) and kept separately as <code>upper_bound_weather_only</code>.
        </p>
        <TemperatureCurve temperature={temps} />
      </Chapter>

      <Chapter id="zones" no="06" eyebrow="Zones" title="Zone load, 2018 vs 2025">
        <p>
          Far West mean load went from {fmtGW(farWest.mw2018)} to {fmtGW(farWest.mw2025)}, <strong>+{farWest.growthPct.toFixed(0)}%</strong>, the largest change of the {meta.zones} zones.
          Its share of system load went from {fwShare(0).toFixed(1)}% to {fwShare(zones.years.length - 1).toFixed(1)}%.
        </p>
        <div className="grid-2">
          <ZoneGrowth zones={zones} />
          <ZoneShare zones={zones} />
        </div>
      </Chapter>

      <Chapter id="uri" no="07" eyebrow="Extremes" title="Winter Storm Uri, February 2021">
        <p>
          The <code>uri_flag</code> covers {meta.uriHours} hours (Feb 14–20, 2021). Load peaked at <strong>{fmtGW(uriPeak.mw)}</strong> ({fmtHour(uriPeak.t)}).
          Within the next 48 hours it fell to <strong>{fmtGW(uriAfter.mw)}</strong> ({fmtHour(uriAfter.t)}). The lowest zone-average temperature was {uriColdest.temp.toFixed(1)} °F ({fmtHour(uriColdest.t)}).
        </p>
        <UriPanels uri={uri} />
      </Chapter>

      <Chapter id="tail" no="08" eyebrow="The tail" title="Load distribution">
        <p>
          Median hourly load: {fmtGW(distribution.p50)}. 99th percentile: <strong>{fmtGW(distribution.p99)}</strong>. Maximum: {fmtGW(distribution.max)}.
          The five highest hours are all in July–August 2026.
        </p>
        <div className="grid-2 wide-left">
          <LoadHistogram distribution={distribution} />
          <Figure title="The five highest hours" sub="System load, all 77,496 hours ranked">
            <ol className="ranked">
              {distribution.topHours.map((h, i) => (
                <li key={h.at}><span className="rank">{i + 1}</span><span>{fmtHour(h.at)}, {h.at.slice(0, 4)}</span><span className="mw">{fmtInt(h.mw)} MW</span></li>
              ))}
            </ol>
          </Figure>
        </div>
      </Chapter>

      <Chapter id="splits" no="09" eyebrow="Splits" title="Train, validation, test">
        <p>
          Chronological splits, no shuffling: train {train.start} → {train.end}, validation {splits[1].start} → {splits[1].end}, test {test.start} → {test.end}.
          Imputers and scalers are fit on train only. Test peak: <strong>{fmtGW(test.peak)}</strong>. Train peak: {fmtGW(train.peak)}. Difference: +{fmtGW(test.peak - train.peak)}.
        </p>
        <Figure
          title="The 77,496 hours, in time order"
          sub="Share of hours in each chronological split"
          legend={<div className="legend">{splits.map(s => <span key={s.split}><i className="box" style={{ background: SPLIT_COLOR[s.split] }} />{SPLIT_LABEL[s.split]}</span>)}</div>}
        >
          <SplitBar splits={splits} />
          <table className="data">
            <thead><tr><th>Split</th><th>Period</th><th>Hours</th><th>Day-ahead samples</th><th>Mean load</th><th>Peak load</th></tr></thead>
            <tbody>
              {splits.map(s => (
                <tr key={s.split}>
                  <td><i className="swatch" style={{ background: SPLIT_COLOR[s.split] }} />{SPLIT_LABEL[s.split]}</td>
                  <td>{s.start} → {s.end}</td>
                  <td>{fmtInt(s.hours)} ({((s.hours / total) * 100).toFixed(0)}%)</td>
                  <td>{s.days != null ? fmtInt(s.days) : '—'}</td>
                  <td>{fmtGW(s.mean)}</td>
                  <td>{fmtGW(s.peak)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Figure>
        <SplitRange splits={splits} />
      </Chapter>

      <Chapter id="next" no="10" eyebrow="Coming next" title="Steps 1–6">
        <p>Remaining steps from <code>MASTER_PLAN.md</code>. Status: planned.</p>
        <ol className="steps">
          {NEXT_STEPS.map(([step, title, body]) => (
            <li key={step}><span className="label">{step}</span><strong>{title}</strong><span>{body}</span></li>
          ))}
        </ol>
      </Chapter>
    </main>
  );
}
