// Strictly linear, monotonic 00 -> N navigation rail for Read Mode.
import { colorForPart } from './StopView';

const pad = n => String(n).padStart(2, '0');

export default function StoryRail({ parts, stops, index, onGo }) {
  return (
    <nav className="story-rail" aria-label="Slides">
      <button
        type="button"
        className={`rail-overview ${index === 0 ? 'active' : ''}`}
        onClick={() => onGo(0)}
      >
        <span className="rail-dot" />
        <span className="rail-ov-label">00 · Overview</span>
      </button>

      <div className="rail-parts">
        {parts.map(part => {
          const partStops = stops
            .map((s, idx) => ({ stop: s, num: idx + 1 }))
            .filter(item => item.stop.part === part.id);
          const isPartActive = partStops.some(item => item.num === index);

          return (
            <div
              key={part.id}
              className={`rail-part ${isPartActive ? 'part-active' : ''}`}
              style={{ '--c': colorForPart(part) }}
            >
              <div className="rail-part-head">
                <span className="rail-part-tag">P{part.id}</span>
                <span className="rail-part-title">{part.title}</span>
              </div>
              <ol className="rail-list">
                {partStops.map(({ stop, num }) => {
                  const active = num === index;
                  const visited = num < index;
                  return (
                    <li key={stop.id}>
                      <button
                        type="button"
                        className={`rail-item ${active ? 'active' : visited ? 'visited' : ''}`}
                        onClick={() => onGo(num)}
                      >
                        <span className="rail-num">{pad(num)}</span>
                        <span className="rail-title">{stop.title}</span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </div>
          );
        })}
      </div>
    </nav>
  );
}
