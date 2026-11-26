// Second-window Presenter View (?presenter=1), synced in real time to the projected tab via BroadcastChannel.
import { useEffect, useState } from 'react';
import { colorForPart } from './StopView';

const pad = n => String(n).padStart(2, '0');
const STEP_NAMES = ['Stats', 'Chart', 'Text'];

function fmtClock(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${pad(m)}:${pad(s)}`;
}

export default function PresenterView({ story, state, onCommand }) {
  const stops = story.stops || [];
  const { index, reveal, present, theme } = state;
  const current = index > 0 ? stops[index - 1] : null;
  const next = index < stops.length ? stops[index] : null;
  const currentPart = current ? story.parts.find(p => p.id === current.part) : null;

  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(true);
  const [nowStr, setNowStr] = useState(() => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

  const noteKey = current ? `beadline-notes-${current.id}` : 'beadline-notes-intro';
  const defaultNotes = current ? (current.notes?.text || '') : (story.opening?.text || '');
  const [noteText, setNoteText] = useState(() => localStorage.getItem(noteKey) ?? defaultNotes);

  useEffect(() => {
    setNoteText(localStorage.getItem(noteKey) ?? defaultNotes);
  }, [noteKey, defaultNotes]);

  useEffect(() => {
    const id = setInterval(() => {
      if (running) setElapsed(s => s + 1);
      setNowStr(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }, 1000);
    return () => clearInterval(id);
  }, [running]);

  return (
    <div className="presenter-shell" style={{ '--p': colorForPart(currentPart) }}>
      <header className="presenter-top">
        <div className="presenter-brand">
          <b>{story.project?.brand || 'Beadline'}</b> · Speaker View
          <span className="presenter-sync-pill">Linked to main window</span>
        </div>

        <div className="presenter-clock-box">
          <span className="presenter-timer">{fmtClock(elapsed)}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setRunning(r => !r)}>
            {running ? 'Pause' : 'Resume'}
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => setElapsed(0)}>
            Reset
          </button>
          <span className="presenter-wall-clock">{nowStr}</span>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => onCommand({ type: 'toggle-theme' })}
          >
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      <div className="presenter-body">
        <section className="presenter-current l1">
          <div className="presenter-scene-head">
            {current ? (
              <>
                <span className="part-chip">Part {currentPart?.id} · {currentPart?.title}</span>
                <span className="stop-no">Slide {pad(index)} / {pad(stops.length)} · {current.title}</span>
              </>
            ) : (
              <span className="part-chip">00 · Overview</span>
            )}
            <span className="presenter-mode-tag">
              {present ? `Present mode · part ${reveal}/3 (${STEP_NAMES[Math.max(0, reveal - 1)]})` : 'Read mode (everything shown)'}
            </span>
          </div>

          <h2>{current ? current.headline : story.title}</h2>

          {current?.say && (
            <div className="presenter-say">
              <span className="field-label">MAIN POINT</span>
              <p>{current.say}</p>
            </div>
          )}

          <div className="presenter-sub-box">
            <span className="field-label">SLIDE TEXT (HIDDEN ON SCREEN IN PRESENT MODE)</span>
            <p>{current ? current.sub : story.intro?.what}</p>
          </div>

          <div className="presenter-notes-box">
            <span className="field-label">SPEAKER NOTES (YOU CAN EDIT)</span>
            <textarea
              value={noteText}
              onChange={e => {
                setNoteText(e.target.value);
                localStorage.setItem(noteKey, e.target.value);
              }}
              rows={5}
            />
          </div>
        </section>

        <aside className="presenter-side">
          <div className="presenter-controls-card">
            <span className="field-label">SCREEN CONTROLS</span>
            <div className="presenter-nav-btns">
              <button type="button" className="btn btn-secondary" onClick={() => onCommand({ type: 'step-prev' })}>
                ← Back
              </button>
              <button type="button" className="btn btn-primary" onClick={() => onCommand({ type: 'step-next' })}>
                Next →
              </button>
            </div>
            <div className="presenter-nav-btns">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => onCommand({ type: 'toggle-present' })}
              >
                {present ? 'Switch screen to Read mode' : 'Switch screen to Present mode (F)'}
              </button>
            </div>
          </div>

          <div className="presenter-next-card">
            <span className="field-label">UP NEXT</span>
            {next ? (
              <>
                <div className="presenter-next-title">Slide {pad(index + 1)} · {next.title}</div>
                <p className="presenter-next-hl">{next.headline}</p>
                {next.stats && (
                  <ul className="presenter-next-stats">
                    {next.stats.map(([v, l]) => <li key={v + l}><b>{v}</b> — {l}</li>)}
                  </ul>
                )}
              </>
            ) : (
              <p className="presenter-next-hl">Last slide. Next: questions.</p>
            )}
          </div>

          <div className="presenter-jump-card">
            <span className="field-label">GO TO SLIDE</span>
            <div className="presenter-jump-grid">
              <button
                type="button"
                className={`chip ${index === 0 ? 'chip-active' : ''}`}
                onClick={() => onCommand({ type: 'goto', index: 0 })}
              >
                00 · Overview
              </button>
              {stops.map((s, i) => (
                <button
                  key={s.id}
                  type="button"
                  className={`chip ${index === i + 1 ? 'chip-active' : ''}`}
                  onClick={() => onCommand({ type: 'goto', index: i + 1 })}
                >
                  {pad(i + 1)} · {s.title}
                </button>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
