// Start-Here overview: driven entirely by `story` (exported from beadline_project.py + export_story_data.py).
import { useMemo, useState } from 'react';
import { matchBead } from '../beadline/ieee754';
import { StoryChart } from './charts';
import { colorForPart } from './StopView';

const pad = n => String(n).padStart(2, '0');

export default function Overview({ story, beads, eda, onGo, onPick, onOpenMachine, onStartPresent }) {
  const { intro, parts, project } = story;
  const [query, setQuery] = useState('');
  const matches = useMemo(
    () => (query.trim() ? beads.filter(b => matchBead(b, query).matched).slice(0, 12) : []),
    [beads, query],
  );
  const partById = useMemo(() => new Map(parts.map(p => [p.id, p])), [parts]);
  const accent2 = colorForPart(partById.get(2) || parts[0]);
  const defaultBead = project?.defaultBead || beads[0]?.name;

  return (
    <section className="overview">
      <header className="ov-hero">
        <h1>{story.title}</h1>
        <p className="ov-byline">{story.byline}</p>
        <p className="ov-what">{intro.what}</p>
        <div className="ov-cta-row">
          <button type="button" className="btn btn-primary ov-start" onClick={() => onGo(1)}>
            Start the Slides →
          </button>
          {onStartPresent && (
            <button type="button" className="btn btn-secondary ov-start" onClick={onStartPresent}>
              Present (F)
            </button>
          )}
          <button type="button" className="btn btn-secondary ov-start" onClick={() => onOpenMachine(defaultBead)}>
            View Code Steps ({beads.length})
          </button>
        </div>
      </header>

      <div className="ov-pipeline" aria-label="Project pipeline">
        {intro.pipeline.map(([title, text, partId], i) => (
          <div key={title} className="ov-step-wrap">
            <div className="l1 ov-step" style={{ '--p': colorForPart(partById.get(partId)) }}>
              <span className="flow-n">{i + 1}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
            {i < intro.pipeline.length - 1 && <span className="ov-arrow" aria-hidden="true">→</span>}
          </div>
        ))}
      </div>

      <div className="ov-answer l1" style={{ '--p': accent2 }}>
        <h2>Main Results</h2>
        <div className="stats">
          {intro.answer.map(([value, label]) => (
            <div key={label} className="l3 stat"><b>{value}</b><span>{label}</span></div>
          ))}
        </div>
        <div className="l2 chart-card"><StoryChart chart={intro.chart} eda={eda} source={project?.source} /></div>
      </div>

      <div className="ov-two">
        <div className="l1 ov-box" style={{ '--p': '#6B717D' }}>
          <h2>Terms</h2>
          <dl className="ov-terms">
            {intro.terms.map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}
          </dl>
        </div>
        <div className="l1 ov-box" style={{ '--p': '#6B717D' }}>
          <h2>How to Use This Page</h2>
          <dl className="ov-terms">
            {intro.howto.map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}
          </dl>
        </div>
      </div>

      <div className="ov-toc-head">
        <h2 className="ov-section">All {story.stops.length} Slides</h2>
        <input
          className="token-input ov-search"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search a step, model, or metric (e.g. CG, 1,691.65, Uri)..."
        />
      </div>
      {matches.length > 0 && (
        <div className="quick-chips">
          {matches.map(b => (
            <button key={b.name} type="button" className="chip" onClick={() => onPick(b.name)}>
              {b.title || b.name}
            </button>
          ))}
        </div>
      )}

      <div className="parts">
        {parts.map(p => {
          const count = story.stops.filter(s => s.part === p.id).length;
          return (
            <div key={p.id} className="l1 part-panel" style={{ '--p': colorForPart(p) }}>
              <div className="part-head">
                <span className="part-chip">Part {p.id}</span>
                <h2>{p.title}</h2>
                <span className="part-speaker">{count} {count === 1 ? 'slide' : 'slides'}</span>
              </div>
              <p className="part-q">{p.question}</p>
              <ol className="toc">
                {story.stops.map((s, i) => (s.part !== p.id ? null : (
                  <li key={s.id}>
                    <button type="button" className="l2" onClick={() => onGo(i + 1)}>
                      <span className="n">{pad(i + 1)}</span>
                      <span className="t">{s.title}</span>
                      <span className="h">{s.headline}</span>
                    </button>
                  </li>
                )))}
              </ol>
            </div>
          );
        })}
      </div>
    </section>
  );
}
