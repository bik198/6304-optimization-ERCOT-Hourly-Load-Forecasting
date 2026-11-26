// Beadline: Generic Two-Mode + Present Explorer
//   - Read Mode    : linear 00 -> N sidebar rail + full context paragraph + Proof drawers
//   - Present Mode : (F) hides sidebar, moves paragraph to Notes, fits chart above fold, click-to-reveal (1->2->3)
//   - Presenter    : (P or ?presenter=1) second window with timer, Say-out-loud cue, notes & next scene via BroadcastChannel
//   - Machine Mode : (M) full-screen horizontal necklace strands + bundled cables + scene focus
//   - Depth Zoom   : (↓ / ↑) Story/DAG -> Code -> Numbers -> 64-bit IEEE-754 Bits
import { Component, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import CodeLayer from './beadline/CodeLayer';
import NumbersLayer from './beadline/NumbersLayer';
import BinaryLayer from './beadline/BinaryLayer';
import MachineCanvas from './beadline/MachineCanvas';
import StoryRail from './story/StoryRail';
import StopView, { colorForPart } from './story/StopView';
import Overview from './story/Overview';
import PresenterView from './story/PresenterView';
import PrintDeck from './story/PrintDeck';

const pad = n => String(n).padStart(2, '0');
const CHANNEL_NAME = 'beadline-sync';

class StopBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error) { console.error(error); }
  render() {
    if (this.state.error) {
      return <div className="stop-error">This view failed to render: {String(this.state.error.message || this.state.error)}</div>;
    }
    return this.props.children;
  }
}

function useJson(name) {
  const inline = typeof window !== 'undefined' ? window.__BEADLINE_INLINE_DATA__?.[name] : null;
  const [value, setValue] = useState(inline || null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (inline) return;
    fetch(`${import.meta.env.BASE_URL}data/${name}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setValue)
      .catch(e => setError(e.message));
  }, [name, inline]);
  return [value, error];
}

export default function App() {
  const [beadData, beadErr] = useJson('beads.json');
  const [story, storyErr] = useJson('story.json');
  const [eda] = useJson('eda.json');

  const params = new URLSearchParams(window.location.search);
  const isPresenterWindow = params.get('presenter') === '1';
  const isPrintDeck = params.get('print') === '1';

  const [mode, setMode] = useState(() => (params.get('mode') === 'machine' ? 'machine' : 'story'));
  const [index, setIndex] = useState(Number(params.get('stop')) || 0);
  const [depth, setDepth] = useState(Number(params.get('depth')) || 0);
  const [present, setPresent] = useState(params.get('present') === '1');
  const [reveal, setReveal] = useState(() => {
    const stepParam = Number(params.get('step'));
    if (stepParam >= 1 && stepParam <= 3) return stepParam;
    return params.get('present') === '1' ? 1 : 3;
  });
  const [drillBead, setDrillBead] = useState(null);
  const [focusScene, setFocusScene] = useState(null);
  const [mapGroup, setMapGroup] = useState(null); // Code Steps: null = group map, else the open group
  const [showNotes, setShowNotes] = useState(params.get('notes') === '1');
  const [inspected, setInspected] = useState({ val: 1691.65, label: 'LSTM test MAE' });
  const [theme, setTheme] = useState(() => params.get('theme') || localStorage.getItem('beadline-theme') || 'dark');
  const channelRef = useRef(null);

  useEffect(() => {
    if (!isPrintDeck) {
      document.documentElement.dataset.theme = theme;
      localStorage.setItem('beadline-theme', theme);
    }
  }, [theme, isPrintDeck]);

  useEffect(() => {
    if (story?.project?.docTitle) document.title = story.project.docTitle;
    if (!drillBead && story?.project?.defaultBead) setDrillBead(story.project.defaultBead);
  }, [story, drillBead]);

  const stops = story?.stops || [];
  const parts = story?.parts || [];
  const partColorFn = useCallback(
    partId => colorForPart(parts.find(p => p.id === partId)),
    [parts],
  );

  const stop = mode === 'story' && index > 0 ? stops[index - 1] : null;
  const part = stop ? parts.find(p => p.id === stop.part) : null;
  const defaultBeadName = story?.project?.defaultBead || beadData?.beads?.[0]?.name;
  const activeBeadName = mode === 'machine'
    ? (drillBead || defaultBeadName)
    : (depth > 0 && drillBead ? drillBead : stop?.bead || null);
  const activeBead = beadData?.beads.find(b => b.name === activeBeadName);
  const canDrill = mode === 'machine' ? !!activeBead : !!(stop?.bead || (depth > 0 && activeBead));
  const strandOf = useCallback(name => beadData?.beads.find(b => b.name === name)?.strand ?? null, [beadData]);

  const goStop = useCallback((i, initialReveal) => {
    setMode('story');
    const clamped = Math.max(0, Math.min(stops.length, i));
    setIndex(clamped);
    setDepth(0);
    setReveal(initialReveal ?? (present && clamped > 0 ? 1 : 3));
    if (clamped > 0 && stops[clamped - 1]?.bead) {
      setDrillBead(stops[clamped - 1].bead);
    }
  }, [stops, present]);

  const stepNext = useCallback(() => {
    if (present && index > 0 && reveal < 3) {
      setReveal(r => Math.min(3, r + 1));
    } else if (index < stops.length) {
      goStop(index + 1, present ? 1 : 3);
    }
  }, [present, index, reveal, stops.length, goStop]);

  const stepPrev = useCallback(() => {
    if (present && index > 0 && reveal > 1) {
      setReveal(r => Math.max(1, r - 1));
    } else if (index > 0) {
      goStop(index - 1, present && index - 1 > 0 ? 3 : 3);
    }
  }, [present, index, reveal, goStop]);

  const togglePresent = useCallback(() => {
    setPresent(prev => {
      const next = !prev;
      setMode('story');
      setDepth(0);
      if (next) {
        if (index === 0 && stops.length > 0) {
          setIndex(1);
          setReveal(1);
        } else {
          setReveal(1);
        }
      } else {
        setReveal(3);
      }
      return next;
    });
  }, [index, stops.length]);

  const openMachineFromScene = useCallback((sceneStop, sceneNum, color) => {
    const members = sceneStop.beads?.length ? sceneStop.beads : (sceneStop.bead ? [sceneStop.bead] : []);
    if (sceneStop.bead) setDrillBead(sceneStop.bead);
    setFocusScene({ num: sceneNum, title: sceneStop.title, beads: members, color });
    setMapGroup(strandOf(sceneStop.bead));
    setMode('machine');
    setDepth(0);
  }, [strandOf]);

  const openMachineBead = useCallback(beadName => {
    if (beadName) setDrillBead(beadName);
    else if (stop?.bead) setDrillBead(stop.bead);
    if (stop && index > 0) {
      const members = stop.beads?.length ? stop.beads : (stop.bead ? [stop.bead] : []);
      setFocusScene({ num: index, title: stop.title, beads: members, color: colorForPart(part) });
      setMapGroup(strandOf(beadName || stop.bead));
    } else {
      setFocusScene(null);
      setMapGroup(null);
    }
    setMode('machine');
    setDepth(0);
  }, [stop, index, part, strandOf]);

  const pickBeadFromOverview = useCallback(name => {
    const own = stops.findIndex(s => s.bead === name || (s.beads || []).includes(name));
    if (own >= 0) {
      goStop(own + 1, 3);
      return;
    }
    setFocusScene(null);
    setDrillBead(name);
    setMapGroup(strandOf(name));
    setMode('machine');
    setDepth(0);
  }, [stops, goStop, strandOf]);

  const setDepthSafe = useCallback(d => {
    const target = Math.max(0, Math.min(3, d));
    if (mode === 'story' && target > 0 && stop?.bead && (!drillBead || depth === 0)) {
      setDrillBead(stop.bead);
    }
    if (target === 0 || canDrill) setDepth(target);
  }, [mode, stop, drillBead, depth, canDrill]);

  const openPresenterPopup = useCallback(() => {
    const url = new URL(window.location.href);
    url.searchParams.set('presenter', '1');
    window.open(url.toString(), 'beadline-presenter', 'width=1020,height=720');
  }, []);

  // Sync projected tab <-> Presenter second window via BroadcastChannel
  const stateRef = useRef({ index, reveal, present, theme, mode });
  stateRef.current = { index, reveal, present, theme, mode };

  useEffect(() => {
    if (typeof BroadcastChannel === 'undefined') return undefined;
    const ch = new BroadcastChannel(CHANNEL_NAME);
    channelRef.current = ch;

    ch.onmessage = ev => {
      const msg = ev.data || {};
      if (isPresenterWindow) {
        if (msg.type === 'state') {
          setIndex(msg.index);
          setReveal(msg.reveal);
          setPresent(msg.present);
          setTheme(msg.theme);
        } else if (msg.type === 'hello-ack') {
          setIndex(msg.index);
          setReveal(msg.reveal);
          setPresent(msg.present);
          setTheme(msg.theme);
        }
      } else {
        if (msg.type === 'hello') {
          ch.postMessage({ type: 'hello-ack', ...stateRef.current });
        } else if (msg.type === 'cmd') {
          const { cmd } = msg;
          if (cmd.type === 'step-next') stepNext();
          else if (cmd.type === 'step-prev') stepPrev();
          else if (cmd.type === 'goto') goStop(cmd.index, stateRef.current.present && cmd.index > 0 ? 1 : 3);
          else if (cmd.type === 'toggle-present') togglePresent();
          else if (cmd.type === 'toggle-theme') setTheme(t => (t === 'light' ? 'dark' : 'light'));
        }
      }
    };

    if (isPresenterWindow) {
      ch.postMessage({ type: 'hello' });
    }
    return () => ch.close();
  }, [isPresenterWindow, stepNext, stepPrev, goStop, togglePresent]);

  useEffect(() => {
    if (!isPresenterWindow && channelRef.current) {
      channelRef.current.postMessage({ type: 'state', index, reveal, present, theme });
    }
  }, [isPresenterWindow, index, reveal, present, theme]);

  useEffect(() => {
    if (isPresenterWindow || isPrintDeck) return undefined;
    const onKey = e => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
        if (e.key === 'Escape') document.activeElement.blur();
        return;
      }
      if (e.key === 'f' || e.key === 'F') togglePresent();
      else if (e.key === 'p' || e.key === 'P') openPresenterPopup();
      else if (e.key === 'm' || e.key === 'M') {
        if (mode === 'story') openMachineBead(stop?.bead || drillBead);
        else { setMode('story'); setDepth(0); }
      } else if (mode === 'story' && (e.key === 'ArrowRight' || e.key === ' ')) stepNext();
      else if (mode === 'story' && e.key === 'ArrowLeft') stepPrev();
      else if (mode === 'machine' && depth === 0 && !mapGroup && (e.key === 'ArrowDown' || e.key === 'Enter')) {
        setMapGroup(activeBead?.strand || null);
      } else if ((e.key === 'ArrowDown' || e.key === 'Enter') && canDrill && (mode === 'machine' || index > 0)) {
        setDepthSafe(depth + 1);
      } else if (e.key === 'ArrowUp' || e.key === 'Escape') {
        if (depth > 0) setDepthSafe(depth - 1);
        else if (mode === 'machine' && mapGroup) setMapGroup(null);
        else if (mode === 'machine') setMode('story');
        else if (present) { setPresent(false); setReveal(3); }
      } else if (e.key === 'n' || e.key === 'N') setShowNotes(v => !v);
      else if (e.key === 't' || e.key === 'T') setTheme(t => (t === 'light' ? 'dark' : 'light'));
      else if (e.key === 'Home') goStop(0, 3);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isPresenterWindow, isPrintDeck, mode, index, depth, present, drillBead, stop, canDrill, goStop, stepNext, stepPrev, togglePresent, openPresenterPopup, openMachineBead, setDepthSafe, mapGroup, activeBead]);

  const error = beadErr || storyErr;
  if (!beadData || !story) {
    return (
      <main className="loading">
        {error ? `Missing data (${error}). Run: python export_beads_data.py; python export_story_data.py` : 'Loading…'}
      </main>
    );
  }

  if (isPrintDeck) {
    return <PrintDeck story={story} eda={eda} />;
  }

  if (isPresenterWindow) {
    return (
      <PresenterView
        story={story}
        state={{ index, reveal, present, theme }}
        onCommand={cmd => {
          if (channelRef.current) channelRef.current.postMessage({ type: 'cmd', cmd });
          // Also update local preview state if opened standalone
          if (cmd.type === 'goto') setIndex(cmd.index);
          else if (cmd.type === 'toggle-theme') setTheme(t => (t === 'light' ? 'dark' : 'light'));
        }}
      />
    );
  }

  const prevLabel = mode === 'story' && index > 0 ? (index === 1 ? 'Overview' : `${pad(index - 1)} · ${stops[index - 2].title}`) : null;
  const nextLabel = mode === 'story' && index < stops.length ? `${pad(index + 1)} · ${stops[index].title}` : null;
  const depthLabels = mode === 'machine' ? ['Map', 'Code', 'Numbers', 'Bits'] : ['Slide', 'Code', 'Numbers', 'Bits'];
  const setNumber = (val, label) => setInspected({ val, label });
  const brand = story.project?.brand || 'Beadline';
  const showSidebar = mode === 'story' && !present;

  return (
    <div className={`bl-shell ${present ? 'bl-present' : ''}`} style={{ '--p': stop ? colorForPart(part) : 'var(--btn-primary-bg)' }}>
      <header className="bl-top">
        <button type="button" className="brand" onClick={() => { setPresent(false); goStop(0, 3); }}>
          {brand}<span className="dot">.</span>
        </button>

        <div className="mode-toggle" role="tablist" aria-label="View mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'story' && !present}
            className={mode === 'story' && !present ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { setPresent(false); setReveal(3); setMode('story'); setDepth(0); }}
          >
            Slides · {stops.length}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'story' && present}
            className={mode === 'story' && present ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { if (!present) togglePresent(); else { setMode('story'); setDepth(0); } }}
            title="Full-screen layout; each part appears on click (F)"
          >
            Present (F)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'machine'}
            className={mode === 'machine' ? 'mode-btn active' : 'mode-btn'}
            onClick={() => openMachineBead(stop?.bead || drillBead)}
            title="Show all code steps (M)"
          >
            Code Steps · {beadData.beads.length}
          </button>
        </div>

        <nav className="crumbs" aria-label="You are here">
          {mode === 'story' ? (
            <>
              <button type="button" onClick={() => goStop(0, 3)}>Overview</button>
              {part && <><i>›</i><span className="crumb-part">Part {part.id} · {part.title}</span></>}
              {stop && <><i>›</i><button type="button" onClick={() => setDepthSafe(0)}>{pad(index)} · {stop.title}</button></>}
              {stop && depth > 0 && <><i>›</i><code>{activeBeadName}</code><i>›</i><b>{depthLabels[depth]}</b></>}
            </>
          ) : (
            <>
              <button type="button" onClick={() => { setDepthSafe(0); setMapGroup(null); }}>Code Steps</button>
              {(mapGroup || depth > 0) && activeBead && (
                <>
                  <i>›</i>
                  <button type="button" onClick={() => { setDepthSafe(0); setMapGroup(activeBead.strand); }}>
                    {beadData.strands.find(s => s.id === activeBead.strand)?.label || activeBead.strand}
                  </button>
                  <i>›</i><span>{activeBead.title || activeBead.name}</span>
                </>
              )}
              {depth > 0 && <><i>›</i><b>{depthLabels[depth]}</b></>}
            </>
          )}
        </nav>

        <div className="top-right-tools">
          {mode === 'story' && <span className="stop-counter">{index} / {stops.length}</span>}
          <button
            type="button"
            className="theme-btn"
            onClick={openPresenterPopup}
            title="Open speaker notes in a second window (P)"
          >
            Speaker View
          </button>
          <button
            type="button"
            className="theme-btn"
            onClick={() => setTheme(t => (t === 'light' ? 'dark' : 'light'))}
            title="Switch theme (T)"
          >
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      <div className={showSidebar ? 'bl-body' : 'bl-body bl-body-full'}>
        {showSidebar && (
          <aside className="bl-map">
            <StoryRail
              parts={parts}
              stops={stops}
              index={index}
              onGo={i => goStop(i, 3)}
            />
          </aside>
        )}

        <main
          className={mode === 'machine' && depth === 0 ? 'bl-main bl-main-dag' : present ? 'bl-main bl-main-present' : 'bl-main'}
          key={`${mode}-${index}-${depth}-${depth > 0 ? activeBeadName : ''}`}
        >
          <StopBoundary>
            {mode === 'story' && !stop && (
              <Overview
                story={story}
                beads={beadData.beads}
                eda={eda}
                onGo={i => goStop(i, 3)}
                onPick={pickBeadFromOverview}
                onOpenMachine={name => { setFocusScene(null); setDrillBead(name); setMapGroup(null); setMode('machine'); setDepth(0); }}
                onStartPresent={() => { setPresent(true); setMode('story'); setIndex(1); setReveal(1); setDepth(0); }}
              />
            )}
            {mode === 'story' && stop && depth === 0 && (
              <StopView
                stop={stop}
                part={part}
                number={index}
                eda={eda}
                source={story.project?.source}
                showNotes={showNotes}
                present={present}
                revealStep={reveal}
                onAdvanceReveal={() => setReveal(r => Math.min(3, r + 1))}
                onOpenMachine={openMachineFromScene}
              />
            )}
            {mode === 'machine' && depth === 0 && (
              <MachineCanvas
                data={beadData}
                stops={stops}
                partColor={partColorFn}
                selectedBeadName={activeBeadName}
                onSelectBead={setDrillBead}
                onOpenStop={num => goStop(num, present ? 3 : 3)}
                onOpenDepth={setDepthSafe}
                focusScene={focusScene}
                onClearFocus={() => setFocusScene(null)}
                group={mapGroup}
                onGroupChange={setMapGroup}
              />
            )}
            {depth > 0 && activeBead && (
              <section className="drill" style={{ '--p': stop ? colorForPart(part) : 'var(--btn-primary-bg)' }}>
                {depth === 1 && (
                  <CodeLayer
                    data={beadData}
                    selectedBeadName={activeBead.name}
                    onSelectBead={setDrillBead}
                    onZoomTo={d => setDepthSafe(Math.max(0, d - 1))}
                  />
                )}
                {depth === 2 && (
                  <NumbersLayer
                    data={beadData}
                    selectedBeadName={activeBead.name}
                    onInspectNumber={setNumber}
                    onZoomTo={d => setDepthSafe(Math.max(0, d - 1))}
                  />
                )}
                {depth === 3 && (
                  <BinaryLayer
                    data={beadData}
                    selectedBeadName={activeBead.name}
                    inspectedNumber={inspected}
                    onInspectNumber={setNumber}
                    onZoomTo={d => setDepthSafe(Math.max(0, d - 1))}
                  />
                )}
              </section>
            )}
          </StopBoundary>
        </main>
      </div>

      <footer className="bl-foot">
        {mode === 'story' ? (
          <button type="button" className="nav-btn" disabled={prevLabel == null && !(present && reveal > 1)} onClick={stepPrev}>
            ← {present && reveal > 1 ? `Step ${reveal - 1}` : prevLabel}
          </button>
        ) : (
          <button type="button" className="nav-btn" onClick={() => { setMode('story'); setDepth(0); }}>
            ← Back to Slides
          </button>
        )}

        {(mode === 'machine' || stop) ? (
          <div className="dock" role="toolbar" aria-label="Detail level, slide parts, and notes">
            {present && mode === 'story' && stop && depth === 0 && (
              <>
                <div className="build-steps" role="group" aria-label="Slide part">
                  {['1 · Stats', '2 · Chart', '3 · Text'].map((lbl, idx) => (
                    <button
                      key={lbl}
                      type="button"
                      className={reveal === idx + 1 ? 'on' : reveal > idx + 1 ? 'soft on' : ''}
                      onClick={() => setReveal(idx + 1)}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
                <span className="dock-gap" />
              </>
            )}
            {depthLabels.map((d, i) => (
              <button
                key={d}
                type="button"
                aria-pressed={depth === i}
                disabled={i > 0 && !canDrill}
                className={depth === i ? 'on' : ''}
                onClick={() => setDepthSafe(i)}
              >
                {i > 0 && '↓ '}{d}
              </button>
            ))}
            {mode === 'story' && stop && (
              <>
                <span className="dock-gap" />
                <button
                  type="button"
                  aria-pressed={showNotes}
                  className={showNotes ? 'on soft' : 'soft'}
                  onClick={() => setShowNotes(v => !v)}
                  title="Show or hide speaker notes (N)"
                >
                  Notes
                </button>
              </>
            )}
          </div>
        ) : (
          <span />
        )}

        {mode === 'story' ? (
          <button
            type="button"
            className="nav-btn"
            disabled={nextLabel == null && !(present && index > 0 && reveal < 3)}
            onClick={stepNext}
          >
            {present && index > 0 && reveal < 3 ? `Next: ${['Stats', 'Chart', 'Text'][reveal]} (${reveal + 1}/3)` : nextLabel} →
          </button>
        ) : (
          <span className="keys" style={{ justifySelf: 'end' }}>Scroll up to zoom in · Scroll down to zoom out · ↑ goes back</span>
        )}
      </footer>
    </div>
  );
}
