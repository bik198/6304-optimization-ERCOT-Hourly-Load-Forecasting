// One scene of the guided story:
// - Read mode (default): headline + factual summary + stats + chart + callout + expandable Proof drawer.
// - Present mode (F): moves `sub` into Notes, caps headline at 2 lines, fits chart above the fold,
//   and reveals 1) stats -> 2) chart -> 3) callout one click / key at a time.
import { useEffect, useState } from 'react';
import { StoryChart } from './charts';

export const FALLBACK_PART_COLOR = { 1: '#3987e5', 2: '#199e70', 3: '#c98500', 4: '#d55181', 5: '#9085e9' };
export const colorForPart = part => part?.color || FALLBACK_PART_COLOR[part?.id] || '#3987e5';

function Stats({ stats }) {
  return (
    <div className="stats">
      {stats.map(([value, label]) => (
        <div key={value + label} className="l3 stat"><b>{value}</b><span>{label}</span></div>
      ))}
    </div>
  );
}

function Callouts({ items }) {
  return items.map(([title, text]) => (
    <div key={title} className="l2 card callout"><h3>{title}</h3><p>{text}</p></div>
  ));
}

export function Table({ table }) {
  return (
    <div className="l2 card table-card">
      <table className="deck-table">
        <thead><tr>{table.columns.map(c => <th key={c} className="l3">{c}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className={table.highlight === i ? 'hl' : ''}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {table.source && <div className="card-source">{table.source}</div>}
    </div>
  );
}

function Flow({ steps }) {
  return (
    <div className="l2 card flow">
      {steps.map(([title, text], i) => (
        <div key={title} className="flow-step">
          <div className="l3 flow-box"><span className="flow-n">{i + 1}</span><h3>{title}</h3><p>{text}</p></div>
          {i < steps.length - 1 && <span className="flow-arrow" aria-hidden="true">→</span>}
        </div>
      ))}
    </div>
  );
}

/** Presenter cue + paragraph (when in Present mode) + editable speaker notes. */
function Notes({ stopId, say, sub, showSub, notes }) {
  const key = `beadline-notes-${stopId}`;
  const original = notes?.text || '';
  const [text, setText] = useState(() => localStorage.getItem(key) ?? original);
  useEffect(() => { setText(localStorage.getItem(key) ?? original); }, [key, original]);
  const edited = text !== original;
  return (
    <div className="l2 card notes">
      <div className="notes-head">
        <h3>Speaker Notes</h3>
        {edited && <button type="button" className="link-btn" onClick={() => { localStorage.removeItem(key); setText(original); }}>Reset</button>}
      </div>
      {say && <p className="say-cue"><b>Main point:</b> {say}</p>}
      {showSub && sub && <p className="notes-sub">{sub}</p>}
      <textarea
        value={text}
        placeholder="Type your notes for this slide..."
        onChange={e => { setText(e.target.value); localStorage.setItem(key, e.target.value); }}
        rows={5}
      />
    </div>
  );
}

export default function StopView({
  stop,
  part,
  number,
  eda,
  source,
  showNotes,
  present = false,
  revealStep = 3,
  onAdvanceReveal,
  onOpenMachine,
  printMode = false,
}) {
  const [showDetails, setShowDetails] = useState(false);
  useEffect(() => { setShowDetails(false); }, [stop.id]);

  const charts = stop.charts || [];
  const hasCallouts = (stop.callouts?.length || 0) > 0;
  const side = (hasCallouts ? 1 : 0) + (showNotes && !printMode ? 1 : 0) > 0;
  const hasMain = !!(charts.length || stop.table || stop.flow || stop.formula);
  const layout = !side ? 'single' : !hasMain ? 'row' : charts.some(c => c.type === 'image') ? 'stacked' : '';
  const pColor = colorForPart(part);

  const stepStats = !present || printMode || revealStep >= 1;
  const stepMain = !present || printMode || revealStep >= 2;
  const stepSide = !present || printMode || revealStep >= 3;

  const handleStageClick = e => {
    if (!present || printMode || revealStep >= 3 || !onAdvanceReveal) return;
    if (e.target.closest('button, a, textarea, input, .details-wrap, .notes')) return;
    onAdvanceReveal();
  };

  return (
    <section
      className={`stop l1 ${present ? 'stop-present' : ''} ${printMode ? 'stop-print' : ''}`}
      style={{ '--p': pColor }}
      onClick={handleStageClick}
    >
      <div className="stop-head">
        <span className="part-chip">Part {part.id} · {part.title}</span>
        <span className="stop-no">Slide {String(number).padStart(2, '0')} · {stop.title}</span>
        {stop.bead && onOpenMachine && !printMode && (
          <button
            type="button"
            className="bead-anchor-btn"
            onClick={() => onOpenMachine(stop, number, pColor)}
            title={`Show the ${(stop.beads || [stop.bead]).length} code steps behind this slide`}
          >
            Show Code Steps →
          </button>
        )}
      </div>

      <h1>{stop.headline}</h1>
      {!present && stop.sub && <p className="stop-sub">{stop.sub}</p>}

      {stop.stats && (
        <div className={`reveal-slot ${stepStats ? 'revealed' : 'concealed'}`}>
          <Stats stats={stop.stats} />
        </div>
      )}

      <div className={`stop-grid ${layout}`}>
        {hasMain && (
          <div className={`stop-main reveal-slot ${stepMain ? 'revealed' : 'concealed'}`}>
            {stop.formula && <div className="l2 card formula">{stop.formula}</div>}
            {stop.flow && <Flow steps={stop.flow} />}
            {charts.map((c, i) => (
              <div key={i} className="l2 chart-card">
                <StoryChart chart={c} eda={eda} compact={present || printMode} source={source} />
              </div>
            ))}
            {stop.table && <Table table={stop.table} />}
          </div>
        )}
        {side && (
          <aside className="stop-side">
            {hasCallouts && (
              <div className={`callouts-slot reveal-slot ${stepSide ? 'revealed' : 'concealed'}`}>
                <Callouts items={stop.callouts} />
              </div>
            )}
            {showNotes && !printMode && (
              <Notes stopId={stop.id} say={stop.say} sub={stop.sub} showSub={present} notes={stop.notes} />
            )}
          </aside>
        )}
      </div>

      {stop.details && !printMode && (
        <div className="details-wrap">
          <button
            type="button"
            className={`details-toggle l2 ${showDetails ? 'open' : ''}`}
            onClick={() => setShowDetails(v => !v)}
            aria-expanded={showDetails}
          >
            <span className="details-caret" aria-hidden="true">{showDetails ? '▾' : '▸'}</span>
            <span className="details-label">{showDetails ? 'Hide' : 'Show'} {stop.details.label}</span>
          </button>
          {showDetails && (
            <div className="details-body">
              {stop.details.formula && <div className="l2 card formula">{stop.details.formula}</div>}
              {(stop.details.charts || []).map((c, i) => (
                <div key={i} className="l2 chart-card"><StoryChart chart={c} eda={eda} source={source} /></div>
              ))}
              {(stop.details.tables || []).map((t, i) => (
                <Table key={i} table={t} />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
