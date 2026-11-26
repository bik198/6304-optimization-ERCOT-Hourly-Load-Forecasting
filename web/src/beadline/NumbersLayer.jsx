import { useEffect, useMemo, useState } from 'react';

function fmtVal(v) {
  if (Array.isArray(v)) return `[${v.map(fmtVal).join(', ')}]`;
  if (typeof v === 'object' && v !== null) return JSON.stringify(v);
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', { maximumFractionDigits: 4 });
  return String(v);
}

/**
 * Depth 3 — Live Number Theater.
 * Displays real inputs -> mathematical operation -> real outputs for the active bead,
 * plus an interactive iteration/pass/hour scrubber when trajectory logs exist.
 */
export default function NumbersLayer({
  data,
  selectedBeadName,
  onInspectNumber,
  onZoomTo,
}) {
  const bead = useMemo(
    () => data.beads.find(b => b.name === selectedBeadName) || data.beads[0],
    [data.beads, selectedBeadName],
  );
  const numbers = bead.numbers || {};
  const scrubber = numbers.scrubber || null;
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
  }, [selectedBeadName]);

  useEffect(() => {
    if (!playing || !scrubber) return undefined;
    const id = setInterval(() => {
      setStepIdx(prev => {
        if (prev + 1 >= scrubber.steps.length) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 260);
    return () => clearInterval(id);
  }, [playing, scrubber]);

  const activeStep = scrubber ? scrubber.steps[Math.min(stepIdx, scrubber.steps.length - 1)] : null;

  const inspectableFloats = useMemo(() => {
    const nums = [];
    const pushFrom = list => (list || []).forEach(item => {
      (item.sample || []).forEach(v => {
        if (typeof v === 'number' && Number.isFinite(v)) nums.push({ label: item.label, val: v });
      });
    });
    pushFrom(numbers.inputs);
    pushFrom(numbers.outputs);
    if (activeStep) {
      Object.entries(activeStep.metrics || {}).forEach(([k, v]) => {
        if (typeof v === 'number' && Number.isFinite(v)) nums.push({ label: `${activeStep.label} · ${k}`, val: v });
      });
    }
    return nums;
  }, [numbers, activeStep]);

  return (
    <div className="layer-shell numbers-layer">
      <div className="layer-toolbar">
        <div>
          <h2>{bead.title}</h2>
        </div>
      </div>

      <div className="flow-trio">
        <div className="num-card">
          <span className="eyebrow">In</span>
          {(numbers.inputs || []).map((item, idx) => (
            <div key={idx} className="num-row">
              <div className="num-label">{item.label}</div>
              <div className="num-val">{fmtVal(item.value)} <span className="num-unit">{item.unit}</span></div>
            </div>
          ))}
        </div>

        <div className="num-card num-op-card">
          <span className="eyebrow">Step</span>
          <p className="op-formula">{numbers.operation || bead.explains}</p>
        </div>

        <div className="num-card">
          <span className="eyebrow">Out</span>
          {(numbers.outputs || []).map((item, idx) => (
            <div key={idx} className="num-row">
              <div className="num-label">{item.label}</div>
              <div className="num-val num-val-highlight">{fmtVal(item.value)} <span className="num-unit">{item.unit}</span></div>
            </div>
          ))}
        </div>
      </div>

      {scrubber && activeStep && (
        <div className="scrubber-panel">
          <div className="scrubber-head">
            <h3>{scrubber.title} · <span className="gold-tag">{activeStep.label}</span></h3>
            <div className="toolbar-actions">
              <button type="button" className="btn btn-secondary" onClick={() => setPlaying(p => !p)}>
                {playing ? 'Pause' : 'Play'}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => { setPlaying(false); setStepIdx(0); }}
              >
                Reset
              </button>
            </div>
          </div>

          <input
            type="range"
            min={0}
            max={scrubber.steps.length - 1}
            value={stepIdx}
            onChange={e => { setPlaying(false); setStepIdx(Number(e.target.value)); }}
            className="step-slider"
          />

          <div className="scrubber-metrics">
            {Object.entries(activeStep.metrics || {}).map(([k, v]) => (
              <button
                key={k}
                type="button"
                className="metric-tile"
                onClick={() => {
                  if (typeof v === 'number') {
                    onInspectNumber(v, `${activeStep.label} · ${k}`);
                    onZoomTo(4);
                  }
                }}
              >
                <span className="kpi-label">{k}</span>
                <span className="kpi-value">{fmtVal(v)}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {numbers.table && (
        <div className="num-table-card">
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  {numbers.table.columns.map(col => <th key={col}>{col}</th>)}
                </tr>
              </thead>
              <tbody>
                {numbers.table.rows.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((cell, cIdx) => <td key={cIdx}>{fmtVal(cell)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {inspectableFloats.length > 0 && (
        <div className="bit-picker">
          <span className="field-label">Bits →</span>
          <div className="quick-chips">
            {inspectableFloats.slice(0, 12).map((item, idx) => (
              <button
                key={idx}
                type="button"
                className="chip"
                title={item.label}
                onClick={() => {
                  onInspectNumber(item.val, item.label);
                  onZoomTo(4);
                }}
              >
                {item.val}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
