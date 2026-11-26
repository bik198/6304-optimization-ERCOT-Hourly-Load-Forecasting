import { useEffect, useMemo, useState } from 'react';
import { floatToBits } from './ieee754';

/**
 * Depth 4 — The Silicon Floor (0s and 1s).
 * The punchline depth: shows the actual IEEE-754 64-bit representation of the
 * exact floating-point number you were inspecting at Depth 3.
 */
export default function BinaryLayer({
  data,
  selectedBeadName,
  inspectedNumber,
  onInspectNumber,
  onZoomTo,
}) {
  const bead = useMemo(
    () => data.beads.find(b => b.name === selectedBeadName) || data.beads[0],
    [data.beads, selectedBeadName],
  );
  const [customVal, setCustomVal] = useState(String(inspectedNumber?.val ?? 1691.65));

  useEffect(() => {
    if (inspectedNumber?.val !== undefined) {
      setCustomVal(String(inspectedNumber.val));
    }
  }, [inspectedNumber]);

  const candidates = useMemo(() => {
    const out = [];
    const nums = bead.numbers || {};
    [...(nums.inputs || []), ...(nums.outputs || [])].forEach(item => {
      (item.sample || []).forEach(v => {
        if (typeof v === 'number' && Number.isFinite(v)) out.push({ label: item.label, val: v });
      });
    });
    return out;
  }, [bead]);

  const activeVal = Number.isFinite(Number(customVal)) ? Number(customVal) : (inspectedNumber?.val ?? 1691.65);
  const bits = useMemo(() => floatToBits(activeVal), [activeVal]);

  return (
    <div className="layer-shell binary-layer">
      <div className="layer-toolbar">
        <div>
          <h2>float64</h2>
          <p className="subtitle">{inspectedNumber?.label || bead.title}</p>
        </div>
      </div>

      <div className="bit-hero">
        <div className="bit-input-row">
          <input
            id="bit-float"
            type="number"
            step="any"
            className="token-input token-input-compact"
            value={customVal}
            onChange={e => {
              setCustomVal(e.target.value);
              const n = Number(e.target.value);
              if (Number.isFinite(n)) onInspectNumber(n, 'Custom float64');
            }}
          />
          <code className="hex-pill">{bits.hex}</code>
        </div>

        <div className="ieee-breakdown">
          <div className="bit-part">
            <span className="field-label">Sign</span>
            <div className="bit-stream bit-sign">{bits.sign}</div>
            <span className="muted-note">{bits.sign === '0' ? '+' : '−'}</span>
          </div>
          <div className="bit-part">
            <span className="field-label">Exponent</span>
            <div className="bit-stream bit-exp">{bits.exponent}</div>
            <span className="muted-note">2^{parseInt(bits.exponent, 2) - 1023}</span>
          </div>
          <div className="bit-part bit-part-wide">
            <span className="field-label">Mantissa</span>
            <div className="bit-stream bit-man">{bits.mantissa}</div>
          </div>
        </div>

        <div className="bit-wall">
          {bits.full64.match(/.{1,8}/g).map((byte, idx) => (
            <span key={idx} className="byte-block">{byte}</span>
          ))}
        </div>
      </div>

      {candidates.length > 0 && (
        <div className="bit-picker">
          <div className="quick-chips">
            {candidates.slice(0, 10).map((c, idx) => (
              <button
                key={idx}
                type="button"
                className="chip"
                title={c.label}
                onClick={() => {
                  setCustomVal(String(c.val));
                  onInspectNumber(c.val, c.label);
                }}
              >
                {c.val}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
