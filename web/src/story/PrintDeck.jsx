// Printable / PDF fallback deck (?print=1): Overview cover + 1 page per scene in Light theme.
import { useEffect } from 'react';
import { StoryChart } from './charts';
import StopView, { colorForPart } from './StopView';

export default function PrintDeck({ story, eda }) {
  useEffect(() => {
    document.documentElement.dataset.theme = 'light';
  }, []);

  const { intro, parts, stops, project } = story;
  const partById = new Map(parts.map(p => [p.id, p]));

  return (
    <div className="print-deck">
      <div className="print-toolbar no-print">
        <div>
          <b>{project?.brand || 'Beadline'} Print / PDF</b> — {stops.length} slides, one per landscape page.
        </div>
        <div className="print-toolbar-btns">
          <button type="button" className="btn btn-primary btn-sm" onClick={() => window.print()}>
            Save / Print PDF
          </button>
          <a className="btn btn-secondary btn-sm" href="?">
            ← Back
          </a>
        </div>
      </div>

      {/* Page 0: Cover & Summary */}
      <section className="print-page">
        <div className="l1 ov-answer" style={{ '--p': colorForPart(parts[0]) }}>
          <h1>{story.title}</h1>
          <p className="ov-byline">{story.byline}</p>
          <p className="stop-sub">{intro.what}</p>
          <div className="stats">
            {intro.answer.map(([v, l]) => (
              <div key={l} className="l3 stat"><b>{v}</b><span>{l}</span></div>
            ))}
          </div>
          <div className="l2 chart-card">
            <StoryChart chart={intro.chart} eda={eda} compact source={project?.source} />
          </div>
        </div>
      </section>

      {/* Pages 1..8: All 8 Scenes */}
      {stops.map((s, idx) => (
        <section key={s.id} className="print-page">
          <StopView
            stop={s}
            part={partById.get(s.part)}
            number={idx + 1}
            eda={eda}
            source={project?.source}
            showNotes={false}
            present={false}
            revealStep={3}
            printMode
          />
        </section>
      ))}
    </div>
  );
}
