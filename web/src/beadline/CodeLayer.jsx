import { useMemo } from 'react';

const PY_KEYWORDS = new Set([
  'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'with', 'as',
  'import', 'from', 'class', 'raise', 'try', 'except', 'finally', 'yield',
  'and', 'or', 'not', 'is', 'None', 'True', 'False', 'lambda',
]);

function highlightLine(line) {
  const commentIdx = line.indexOf('#');
  const codePart = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
  const commentPart = commentIdx >= 0 ? line.slice(commentIdx) : '';

  const tokens = codePart.split(/(\b[A-Za-z_][A-Za-z0-9_]*\b|"[^"]*"|'[^']*'|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g);

  return (
    <>
      {tokens.map((tok, i) => {
        if (!tok) return null;
        if (tok.startsWith('"') || tok.startsWith("'")) {
          return <span key={i} className="tok-str">{tok}</span>;
        }
        if (PY_KEYWORDS.has(tok)) {
          return <span key={i} className="tok-kw">{tok}</span>;
        }
        if (/^\d/.test(tok)) {
          return <span key={i} className="tok-num">{tok}</span>;
        }
        if (tok.startsWith('@')) {
          return <span key={i} className="tok-dec">{tok}</span>;
        }
        return <span key={i}>{tok}</span>;
      })}
      {commentPart && <span className="tok-cmt">{commentPart}</span>}
    </>
  );
}

/**
 * Depth 2 — Atomic Source Code.
 * Renders the exact Python lines extracted via `inspect.getsourcelines()` for the active bead.
 */
export default function CodeLayer({ data, selectedBeadName, onSelectBead, onZoomTo }) {
  const bead = useMemo(
    () => data.beads.find(b => b.name === selectedBeadName) || data.beads[0],
    [data.beads, selectedBeadName],
  );
  const lines = useMemo(() => (bead.source || '').split('\n'), [bead.source]);

  return (
    <div className="layer-shell code-layer">
      <div className="layer-toolbar">
        <div>
          <h2>
            {bead.title} <span className="gold-tag">{bead.file}:{bead.line}</span>
          </h2>
        </div>
      </div>

      <div className="code-panel">
        <pre className="code-pre">
          {lines.map((ln, idx) => (
            <div key={idx} className="code-row">
              <span className="line-no">{bead.line + idx}</span>
              <span className="line-txt">{highlightLine(ln)}</span>
            </div>
          ))}
        </pre>
      </div>

      <div className="bead-nav-row">
        <div>
          <div className="quick-chips">
            {(bead.after || []).length === 0 ? (
null
            ) : (
              bead.after.map(u => (
                <button key={u} type="button" className="chip" onClick={() => onSelectBead(u)}>
                  ← {u}
                </button>
              ))
            )}
          </div>
        </div>
        <div>
          <div className="quick-chips">
            {(bead.next || []).length === 0 ? (
null
            ) : (
              bead.next.map(d => (
                <button key={d} type="button" className="chip" onClick={() => onSelectBead(d)}>
                  {d} →
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
