import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { matchBead } from './ieee754';
import './machine.css';

/**
 * Code Steps view, two zoom levels:
 *  1. Group map: the 7 groups of functions and what passes between them (positions from
 *     beadline_project.py STRAND_META `pos`, otherwise computed from the links).
 *  2. Function diagram of one group: its functions left to right, with the steps it gets
 *     input from and sends output to shown as boxes at the edges.
 * Scroll up on a group (or double-click / Open) to zoom in; scroll down or ↑ to zoom out.
 */

const EARLIER = 'var(--text-link)';
const LATER = 'var(--text-gold)';
const ink = c => `color-mix(in srgb, ${c} 78%, var(--text-bright))`;
const pad2 = n => String(n).padStart(2, '0');

function curve(x1, y1, x2, y2) {
  const dx = Math.max(28, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

/** Longest-path column of every node in a DAG. */
function longestPath(names, parentsOf) {
  const memo = new Map();
  const col = n => {
    if (memo.has(n)) return memo.get(n);
    memo.set(n, 0);
    const ps = parentsOf(n);
    const c = ps.length ? 1 + Math.max(...ps.map(col)) : 0;
    memo.set(n, c);
    return c;
  };
  names.forEach(col);
  return memo;
}

function useSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 1100, h: 640 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => {
      if (el.clientWidth > 0 && el.clientHeight > 0) setSize({ w: el.clientWidth, h: el.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/** Turns a run of wheel events into one zoom-in or zoom-out action. */
function useWheelStep(onIn, onOut) {
  const acc = useRef(0);
  const lockUntil = useRef(0);
  return (e, target) => {
    const now = Date.now();
    if (now < lockUntil.current) return;
    acc.current += e.deltaY;
    if (acc.current < -60 && onIn) {
      acc.current = 0;
      lockUntil.current = now + 700;
      onIn(target);
    } else if (acc.current > 60 && onOut) {
      acc.current = 0;
      lockUntil.current = now + 700;
      onOut();
    } else if (Math.abs(acc.current) > 60) {
      acc.current = 0;
    }
  };
}

export default function MachineCanvas({
  data,
  stops,
  partColor,
  selectedBeadName,
  onSelectBead,
  onOpenStop,
  onOpenDepth,
  focusScene = null,
  onClearFocus,
  group = null,
  onGroupChange,
}) {
  const { beads, strands } = data;
  const [vpRef, size] = useSize();
  const [query, setQuery] = useState('');
  const [hoverGroup, setHoverGroup] = useState(null);
  const [hoverBead, setHoverBead] = useState(null);
  const originRef = useRef(null);
  const [anim, setAnim] = useState(null);

  const strandById = useMemo(() => new Map(strands.map(s => [s.id, s])), [strands]);
  const selectedBead = useMemo(
    () => beads.find(b => b.name === selectedBeadName) || beads[0],
    [beads, selectedBeadName],
  );
  const [mapSel, setMapSel] = useState(selectedBead.strand);
  const sceneSet = useMemo(() => (focusScene?.beads ? new Set(focusScene.beads) : null), [focusScene]);

  // ---------------------------------------------------------------- graph model
  const model = useMemo(() => {
    const byName = new Map(beads.map(b => [b.name, b]));
    const links = [];
    beads.forEach(b => (b.after || []).forEach(p => {
      const src = byName.get(p);
      if (src && src.strand !== b.strand) links.push({ from: p, to: b.name, fs: src.strand, ts: b.strand });
    }));
    const edgeMap = new Map();
    links.forEach(l => {
      const key = `${l.fs}->${l.ts}`;
      if (!edgeMap.has(key)) edgeMap.set(key, { key, from: l.fs, to: l.ts, links: [] });
      edgeMap.get(key).links.push(l);
    });
    const members = new Map(strands.map(s => [s.id, beads.filter(b => b.strand === s.id)]));
    return { byName, links, groupEdges: [...edgeMap.values()], members };
  }, [beads, strands]);

  const stopsByBead = useMemo(() => {
    const m = new Map();
    (stops || []).forEach((s, idx) => {
      const list = s.beads?.length ? s.beads : (s.bead ? [s.bead] : []);
      list.forEach(n => {
        if (!m.has(n)) m.set(n, []);
        m.get(n).push({ num: idx + 1, stop: s, isPrimary: s.bead === n });
      });
    });
    return m;
  }, [stops]);

  const slidesByGroup = useMemo(() => {
    const m = new Map();
    strands.forEach(s => {
      const seen = new Map();
      (model.members.get(s.id) || []).forEach(b => (stopsByBead.get(b.name) || []).forEach(x => seen.set(x.num, x)));
      m.set(s.id, [...seen.values()].sort((a, b) => a.num - b.num));
    });
    return m;
  }, [strands, model, stopsByBead]);

  const matches = useMemo(
    () => (query.trim() ? beads.filter(b => matchBead(b, query).matched) : []),
    [beads, query],
  );
  const matchCount = useMemo(() => {
    const m = new Map();
    matches.forEach(b => m.set(b.strand, (m.get(b.strand) || 0) + 1));
    return m;
  }, [matches]);

  // Transitive earlier / later steps of the hovered or selected function.
  const focusName = hoverBead || selectedBead.name;
  const lineage = useMemo(() => {
    const ancestors = new Set();
    const descendants = new Set();
    const up = new Set();
    const down = new Set();
    const walk = (cur, dir) => {
      const b = model.byName.get(cur);
      if (!b) return;
      const nexts = dir === 'up' ? b.after || [] : b.next || [];
      nexts.forEach(n => {
        (dir === 'up' ? up : down).add(dir === 'up' ? `${n}->${cur}` : `${cur}->${n}`);
        const set = dir === 'up' ? ancestors : descendants;
        if (!set.has(n)) { set.add(n); walk(n, dir); }
      });
    };
    walk(focusName, 'up');
    walk(focusName, 'down');
    return { ancestors, descendants, up, down };
  }, [focusName, model]);

  // ---------------------------------------------------------------- level 1: group map
  const mapLayout = useMemo(() => {
    const W = size.w;
    const H = size.h;
    let grid;
    if (strands.every(s => Array.isArray(s.pos))) {
      grid = new Map(strands.map(s => [s.id, { col: s.pos[0], row: s.pos[1] }]));
    } else {
      const cols = longestPath(strands.map(s => s.id), id => model.groupEdges.filter(e => e.to === id).map(e => e.from));
      const used = {};
      grid = new Map(strands.map(s => {
        const c = cols.get(s.id);
        used[c] = (used[c] ?? -1) + 1;
        return [s.id, { col: c, row: used[c] }];
      }));
    }
    const cells = [...grid.values()];
    const nCols = Math.max(...cells.map(p => p.col)) + 1;
    const rMin = Math.min(...cells.map(p => p.row));
    const rMax = Math.max(...cells.map(p => p.row));
    const padX = 24;
    const padTop = focusScene ? 66 : 30;
    const padBot = 44;
    const cardW = Math.max(170, Math.min(250, (W - 2 * padX) / nCols - 70));
    // Cards one row apart in the same column must not touch: (H - pads - cardH) / R >= cardH + 12.
    const R = Math.max(1, rMax - rMin);
    const cardH = Math.max(96, Math.min(128, (H - padTop - padBot - 12 * R) / (R + 1)));
    const xAt = c => (nCols === 1 ? (W - cardW) / 2 : padX + (c * (W - 2 * padX - cardW)) / (nCols - 1));
    const yAt = r => (rMax === rMin ? H / 2 : padTop + cardH / 2 + ((r - rMin) / (rMax - rMin)) * (H - padTop - padBot - cardH));
    const nodes = new Map(strands.map(s => {
      const p = grid.get(s.id);
      return [s.id, { id: s.id, x: xAt(p.col), cy: yAt(p.row), w: cardW, h: cardH }];
    }));
    // A group fed by 3+ groups is drawn tall, so every incoming arrow arrives level.
    nodes.forEach(n => {
      const ins = model.groupEdges.filter(e => e.to === n.id).map(e => nodes.get(e.from).cy);
      if (ins.length >= 3) {
        const lo = Math.min(...ins);
        const hi = Math.max(...ins);
        n.h = Math.max(cardH, hi - lo + 76);
        n.cy = (lo + hi) / 2;
      }
    });
    const edges = model.groupEdges.map(e => {
      const a = nodes.get(e.from);
      const b = nodes.get(e.to);
      const x1 = a.x + a.w;
      const y1 = a.cy;
      const x2 = b.x - 6;
      const y2 = Math.max(b.cy - b.h / 2 + 24, Math.min(b.cy + b.h / 2 - 24, a.cy));
      return { ...e, d: curve(x1, y1, x2, y2) };
    });
    return { nodes, edges, W, H };
  }, [size, strands, model, focusScene]);

  // ---------------------------------------------------------------- level 2: one group's functions
  const groupLayout = useMemo(() => {
    if (!group) return null;
    const inner = model.members.get(group) || [];
    const innerSet = new Set(inner.map(b => b.name));
    const parents = n => (model.byName.get(n)?.after || []).filter(p => innerSet.has(p));
    const col = longestPath(inner.map(b => b.name), parents);
    inner.forEach(b => {
      if (parents(b.name).length) return;
      const kids = (b.next || []).filter(k => innerSet.has(k));
      if (kids.length) col.set(b.name, Math.max(0, Math.min(...kids.map(k => col.get(k))) - 1));
    });

    const uniq = (list, key) => [...new Map(list.map(x => [x[key], x])).values()];
    const inStubs = uniq(model.links.filter(l => l.ts === group).map(l => ({ name: l.from, strand: l.fs })), 'name');
    const outStubs = uniq(model.links.filter(l => l.fs === group).map(l => ({ name: l.to, strand: l.ts })), 'name');
    const innerCols = inner.length ? Math.max(...col.values()) + 1 : 0;
    const off = inStubs.length ? 1 : 0;
    const nCols = innerCols + off + (outStubs.length ? 1 : 0);

    const columns = Array.from({ length: nCols }, () => []);
    inStubs.forEach(s => columns[0].push({ ...s, kind: 'in' }));
    inner.forEach(b => columns[off + col.get(b.name)].push({ name: b.name, strand: group, kind: 'fn' }));
    outStubs.forEach(s => columns[nCols - 1].push({ ...s, kind: 'out' }));

    const edges = [];
    inner.forEach(b => parents(b.name).forEach(p => edges.push({ from: p, to: b.name })));
    model.links.filter(l => l.ts === group).forEach(l => edges.push({ from: l.from, to: l.to, cross: true }));
    model.links.filter(l => l.fs === group).forEach(l => edges.push({ from: l.from, to: l.to, cross: true }));

    // Order each column by the average position of what feeds it (fewer crossings).
    const rank = new Map();
    columns.forEach(items => {
      items.forEach((it, i) => {
        const feeders = edges.filter(e => e.to === it.name && rank.has(e.from)).map(e => rank.get(e.from));
        it.order = feeders.length ? feeders.reduce((a, v) => a + v, 0) / feeders.length : i;
      });
      items.sort((a, b) => a.order - b.order);
      items.forEach((it, i) => rank.set(it.name, i - (items.length - 1) / 2));
    });

    // Fit to the viewport width: a long chain wraps onto the next band instead of scrolling sideways.
    const padX = 22;
    const padTop = focusScene ? 112 : 76;
    const padBot = 44;
    const gapX = 42;
    const cardH = 80;
    const W = size.w;
    // Keep every column on one band if cards can stay at least 132px wide; otherwise wrap at 164px.
    const fitsAt = minW => Math.floor((W - 2 * padX + gapX) / (minW + gapX));
    const perBand = fitsAt(132) >= nCols ? nCols : Math.max(2, Math.min(nCols, fitsAt(164)));
    const nBands = Math.ceil(nCols / perBand);
    const cardW = Math.min(212, (W - 2 * padX - (perBand - 1) * gapX) / perBand);
    const bandGap = 46;
    const bandRows = Array.from({ length: nBands }, (_, k) => Math.max(1, ...columns.slice(k * perBand, (k + 1) * perBand).map(c => c.length)));
    const rowStep = cardH + 26;
    const bandTops = [];
    let yCur = padTop;
    bandRows.forEach(r => { bandTops.push(yCur); yCur += r * rowStep + bandGap; });
    const usedH = yCur - bandGap + padBot;
    const H = Math.max(size.h, usedH);
    const extra = Math.max(0, (H - usedH) / 2);
    const pos = new Map();
    columns.forEach((items, c) => {
      const band = Math.floor(c / perBand);
      const inBand = c % perBand;
      const bandCols = Math.min(perBand, nCols - band * perBand);
      const spanW = bandCols * cardW + (bandCols - 1) * gapX;
      const x0 = nBands === 1 ? (W - spanW) / 2 : padX;
      const x = x0 + inBand * (cardW + gapX);
      const mid = extra + bandTops[band] + (bandRows[band] * rowStep) / 2;
      items.forEach((it, i) => pos.set(it.name, { ...it, band, x, cy: mid + (i - (items.length - 1) / 2) * rowStep, w: cardW, h: cardH }));
    });
    const bandBottom = k => extra + bandTops[k] + bandRows[k] * rowStep;
    const paths = edges.map(e => {
      const a = pos.get(e.from);
      const b = pos.get(e.to);
      let d;
      if (a.band === b.band) d = curve(a.x + a.w, a.cy, b.x - 6, b.cy);
      else {
        const yg = bandBottom(a.band) + bandGap / 2 - 4;
        d = `M ${a.x + a.w} ${a.cy} H ${a.x + a.w + 14} V ${yg} H ${b.x - 18} V ${b.cy} H ${b.x - 6}`;
      }
      return { ...e, key: `${e.from}->${e.to}`, d };
    });
    return { pos, paths, W, H, inner };
  }, [group, model, size, focusScene]);

  // ---------------------------------------------------------------- zoom transitions
  const prevGroup = useRef(group);
  useEffect(() => {
    if (vpRef.current) { vpRef.current.scrollLeft = 0; vpRef.current.scrollTop = 0; }
    if (prevGroup.current === group) return;
    setAnim(group ? 'in' : 'out');
    prevGroup.current = group;
    const t = setTimeout(() => setAnim(null), 320);
    return () => clearTimeout(t);
  }, [group]);

  const openGroup = id => {
    const n = mapLayout.nodes.get(id);
    if (n) originRef.current = { x: n.x + n.w / 2, y: n.cy };
    const inGroup = b => model.byName.get(b)?.strand === id;
    const pick = inGroup(selectedBead.name)
      ? selectedBead.name
      : (focusScene?.beads || []).find(inGroup) || (model.members.get(id) || [])[0]?.name;
    if (pick && pick !== selectedBead.name) onSelectBead(pick);
    setMapSel(id);
    setHoverGroup(null);
    onGroupChange?.(id);
  };
  const closeGroup = () => {
    if (group) setMapSel(group);
    setHoverBead(null);
    onGroupChange?.(null);
  };
  const mapWheel = useWheelStep(id => id && openGroup(id), null);
  const groupWheel = useWheelStep(null, closeGroup);

  const jumpToBead = b => {
    onSelectBead(b.name);
    if (group !== b.strand) {
      const n = mapLayout.nodes.get(b.strand);
      if (n) originRef.current = { x: n.x + n.w / 2, y: n.cy };
      onGroupChange?.(b.strand);
    }
    setQuery('');
  };

  // ---------------------------------------------------------------- render helpers
  const groupLabel = id => strandById.get(id)?.label.replace(/^\d+\s*·\s*/, '') || id;
  const origin = originRef.current;
  const originStyle = origin ? { transformOrigin: `${origin.x}px ${origin.y}px` } : undefined;

  const renderMap = () => {
    const { nodes, edges, W, H } = mapLayout;
    const lit = hoverGroup || null;
    return (
      <div className={`gm-stage ${anim === 'out' ? 'zoom-out-enter' : ''}`} style={{ width: W, height: H, ...originStyle }}>
        <svg className="gm-edges" width={W} height={H} aria-hidden="true">
          <defs>
            <marker id="gm-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--line-strong)' }} />
            </marker>
            <marker id="gm-arrow-on" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--text-link)' }} />
            </marker>
          </defs>
          {edges.map(e => {
            const on = lit && (e.from === lit || e.to === lit);
            const dim = lit && !on;
            return (
              <g key={e.key} className={dim ? 'gm-dim' : ''}>
                <path
                  d={e.d}
                  fill="none"
                  style={{ stroke: on ? 'var(--text-link)' : 'var(--line-strong)' }}
                  strokeWidth={on ? 2.6 : 2}
                  markerEnd={on ? 'url(#gm-arrow-on)' : 'url(#gm-arrow)'}
                >
                  <title>{e.links.map(l => `${model.byName.get(l.from)?.title} → ${model.byName.get(l.to)?.title}`).join('\n')}</title>
                </path>
              </g>
            );
          })}
        </svg>
        {strands.map(s => {
          const n = nodes.get(s.id);
          const members = model.members.get(s.id) || [];
          const slides = slidesByGroup.get(s.id) || [];
          const inScene = sceneSet ? members.filter(b => sceneSet.has(b.name)).length : 0;
          const found = matchCount.get(s.id) || 0;
          const dim = (sceneSet && !inScene) || (query.trim() && !found) || (lit && lit !== s.id
            && !model.groupEdges.some(e => (e.from === lit && e.to === s.id) || (e.to === lit && e.from === s.id)));
          const tall = n.h > 220;
          return (
            <div
              key={s.id}
              role="button"
              tabIndex={0}
              className={`gm-card ${mapSel === s.id ? 'sel' : ''} ${dim ? 'gm-dim' : ''} ${inScene ? 'in-scene' : ''}`}
              style={{ left: n.x, top: n.cy - n.h / 2, width: n.w, height: n.h, '--c': ink(s.color), '--scene': focusScene?.color }}
              onMouseEnter={() => setHoverGroup(s.id)}
              onMouseLeave={() => setHoverGroup(null)}
              onClick={() => setMapSel(s.id)}
              onDoubleClick={() => openGroup(s.id)}
              onKeyDown={e => { if (e.key === 'Enter') { e.stopPropagation(); openGroup(s.id); } }}
              onWheel={e => mapWheel(e, s.id)}
            >
              <span className="gm-title"><i className="gm-dot" />{groupLabel(s.id)}</span>
              <span className="gm-count">
                {members.length} functions
                {inScene > 0 && <b className="gm-scene-count"> · slide {pad2(focusScene.num)}: {inScene}</b>}
                {found > 0 && <b className="gm-found"> · {found} found</b>}
              </span>
              <span className={`gm-sum ${tall ? 'tall' : ''}`}>{s.summary}</span>
              {tall && (
                <ul className="gm-fn-list">
                  {members.map(b => <li key={b.name}>{b.title || b.name}</li>)}
                </ul>
              )}
              <span className="gm-foot">
                <span className="gm-slides">
                  {slides.map(x => (
                    <span key={x.num} className="gm-slide" style={{ background: partColor(x.stop.part) }} title={`Slide ${pad2(x.num)} · ${x.stop.title}`}>
                      {pad2(x.num)}
                    </span>
                  ))}
                </span>
                <button type="button" className="gm-open" onClick={e => { e.stopPropagation(); openGroup(s.id); }}>
                  Open →
                </button>
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderGroup = () => {
    const { pos, paths, W, H } = groupLayout;
    const s = strandById.get(group);
    const groupHasScene = !!sceneSet && groupLayout.inner.some(b => sceneSet.has(b.name));
    return (
      <div className={`gv-stage ${anim === 'in' ? 'zoom-in-enter' : ''}`} style={{ width: W, height: H, ...originStyle }} onWheel={groupWheel}>
        <div className="gv-head" style={{ '--c': ink(s?.color || '#888') }}>
          <button type="button" className="btn btn-secondary btn-sm" onClick={closeGroup}>← All groups</button>
          <span className="gv-title"><i className="gm-dot" />{groupLabel(group)}</span>
          <span className="gv-sum">{groupLayout.inner.length} functions · {s?.summary}</span>
        </div>
        <svg className="gm-edges" width={W} height={H} aria-hidden="true">
          <defs>
            {[['n', 'var(--line-strong)'], ['u', EARLIER], ['d', LATER]].map(([id, c]) => (
              <marker key={id} id={`gv-arrow-${id}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" style={{ fill: c }} />
              </marker>
            ))}
          </defs>
          {paths.map(p => {
            const up = lineage.up.has(p.key);
            const down = lineage.down.has(p.key);
            const c = up ? EARLIER : down ? LATER : 'var(--line-strong)';
            return (
              <path
                key={p.key}
                d={p.d}
                fill="none"
                style={{ stroke: c }}
                strokeWidth={up || down ? 2.6 : 1.8}
                strokeDasharray={p.cross ? '5 4' : undefined}
                markerEnd={`url(#gv-arrow-${up ? 'u' : down ? 'd' : 'n'})`}
              />
            );
          })}
        </svg>
        {[...pos.values()].map(it => {
          const b = model.byName.get(it.name);
          if (!b) return null;
          const stub = it.kind !== 'fn';
          const sel = it.name === selectedBead.name;
          const anc = lineage.ancestors.has(it.name);
          const desc = lineage.descendants.has(it.name);
          const inScene = sceneSet?.has(it.name);
          const primary = (stopsByBead.get(it.name) || []).find(x => x.isPrimary);
          const strand = strandById.get(it.strand);
          const found = query.trim() && matches.some(m => m.name === it.name);
          return (
            <div
              key={it.name}
              role="button"
              tabIndex={0}
              className={`gv-card ${stub ? 'stub' : ''} ${sel ? 'sel' : ''} ${anc ? 'anc' : ''} ${desc ? 'desc' : ''} ${inScene ? 'in-scene' : ''} ${found ? 'found' : ''} ${groupHasScene && !inScene && !sel && !stub ? 'gm-dim' : ''}`}
              style={{ left: it.x, top: it.cy - it.h / 2, width: it.w, height: it.h, '--c': ink(strand?.color || '#888'), '--scene': focusScene?.color }}
              title={`${b.title} — ${b.name}()\n${b.explains || ''}`}
              onMouseEnter={() => setHoverBead(it.name)}
              onMouseLeave={() => setHoverBead(null)}
              onClick={() => (stub ? jumpToBead(b) : onSelectBead(it.name))}
              onDoubleClick={() => { if (!stub) { onSelectBead(it.name); onOpenDepth(1); } }}
              onKeyDown={e => { if (e.key === 'Enter' && stub) { e.stopPropagation(); jumpToBead(b); } }}
            >
              {stub && <span className="gv-stub-tag">{it.kind === 'in' ? 'From' : 'To'} {groupLabel(it.strand)}</span>}
              <span className="gv-name">{b.title || b.name}</span>
              <code className="gv-fn">{b.name}()</code>
              {primary && !stub && (
                <span className="gv-slide" style={{ background: partColor(primary.stop.part) }} title={`Slide ${pad2(primary.num)} · ${primary.stop.title}`}>
                  {pad2(primary.num)}
                </span>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // ---------------------------------------------------------------- inspector
  const linkedStops = stopsByBead.get(selectedBead.name) || [];
  const mapGroup = strandById.get(mapSel) || strands[0];
  const feedsFrom = model.links.filter(l => l.ts === mapGroup.id);
  const feedsTo = model.links.filter(l => l.fs === mapGroup.id);
  const title = n => model.byName.get(n)?.title || n;

  const groupInspector = (
    <aside className="dag-inspector">
      <span className="eyebrow">GROUP</span>
      <h3>{groupLabel(mapGroup.id)}</h3>
      <p className="inspector-desc">{mapGroup.summary}</p>
      <button type="button" className="btn btn-primary" onClick={() => openGroup(mapGroup.id)}>
        Open its {(model.members.get(mapGroup.id) || []).length} functions →
      </button>
      {feedsFrom.length > 0 && (
        <div className="inspector-links">
          <span className="field-label">GETS INPUT FROM</span>
          <div className="quick-chips">
            {feedsFrom.map(l => (
              <button key={l.from + l.to} type="button" className="chip gm-link-chip" onClick={() => setMapSel(l.fs)}>
                <b>← {groupLabel(l.fs)}</b><span>{title(l.from)} → {title(l.to)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      {feedsTo.length > 0 && (
        <div className="inspector-links">
          <span className="field-label">SENDS OUTPUT TO</span>
          <div className="quick-chips">
            {feedsTo.map(l => (
              <button key={l.from + l.to} type="button" className="chip gm-link-chip" onClick={() => setMapSel(l.ts)}>
                <b>{groupLabel(l.ts)} →</b><span>{title(l.from)} → {title(l.to)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      {(slidesByGroup.get(mapGroup.id) || []).length > 0 && (
        <div className="inspector-links">
          <span className="field-label">SLIDES THAT USE THIS GROUP</span>
          <div className="quick-chips">
            {slidesByGroup.get(mapGroup.id).map(({ num, stop }) => (
              <button key={stop.id} type="button" className="chip" onClick={() => onOpenStop(num)}>Slide {pad2(num)} · {stop.title} →</button>
            ))}
          </div>
        </div>
      )}
      <div className="inspector-links">
        <span className="field-label">FUNCTIONS</span>
        <div className="quick-chips">
          {(model.members.get(mapGroup.id) || []).map(b => (
            <button key={b.name} type="button" className="chip" title={`${b.name}()`} onClick={() => { onSelectBead(b.name); openGroup(mapGroup.id); }}>
              {b.title || b.name}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );

  const beadInspector = (
    <aside className="dag-inspector">
      <span className="eyebrow">SELECTED STEP</span>
      <h3>{selectedBead.title}</h3>
      <div className="inspector-meta">
        <code className="code-pill">fn: {selectedBead.name}()</code>
        <code className="code-pill">{selectedBead.file} : L{selectedBead.line}–{selectedBead.end_line}</code>
      </div>
      <p className="inspector-desc">{selectedBead.explains}</p>
      <div className="io-box">
        <div>
          <span className="field-label">TAKES IN</span>
          <span className="io-val">{selectedBead.io?.[0] || '—'}</span>
        </div>
        <span className="io-arrow">→</span>
        <div>
          <span className="field-label">PRODUCES</span>
          <span className="io-val">{selectedBead.io?.[1] || '—'}</span>
        </div>
      </div>
      <div className="inspector-actions">
        <button type="button" className="btn btn-primary" onClick={() => onOpenDepth(1)}>View Code →</button>
        <div className="inspector-btn-row">
          <button type="button" className="btn btn-secondary" onClick={() => onOpenDepth(2)}>Numbers</button>
          <button type="button" className="btn btn-secondary" onClick={() => onOpenDepth(3)}>Bits (0s and 1s)</button>
        </div>
      </div>
      {linkedStops.length > 0 && (
        <div className="inspector-links">
          <span className="field-label">SLIDES THAT USE THIS STEP</span>
          <div className="quick-chips">
            {linkedStops.map(({ num, stop }) => (
              <button key={stop.id} type="button" className="chip" onClick={() => onOpenStop(num)}>Slide {pad2(num)} · {stop.title} →</button>
            ))}
          </div>
        </div>
      )}
      <div className="inspector-links">
        <span className="field-label">CONNECTED STEPS ({lineage.ancestors.size} BEFORE · {lineage.descendants.size} AFTER)</span>
        <div className="quick-chips">
          {(selectedBead.after || []).map(up => (
            <button key={up} type="button" className="chip" title={`${up}()`} onClick={() => jumpToBead(model.byName.get(up))}>
              ← {model.byName.get(up)?.title || up}
            </button>
          ))}
          {(selectedBead.next || []).map(dn => (
            <button key={dn} type="button" className="chip" title={`${dn}()`} onClick={() => jumpToBead(model.byName.get(dn))}>
              {model.byName.get(dn)?.title || dn} →
            </button>
          ))}
        </div>
      </div>
    </aside>
  );

  // ---------------------------------------------------------------- layout
  return (
    <div className="machine-shell">
      <div className="machine-bar">
        <div className="machine-filters gm-search">
          <input
            type="text"
            className="token-input token-input-compact"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={`Search ${beads.length} steps by name, number, or word...`}
          />
          {matches.length > 0 && (
            <div className="gm-results">
              {matches.slice(0, 8).map(b => (
                <button key={b.name} type="button" className="chip" onClick={() => jumpToBead(b)}>
                  <span className="strand-dot" style={{ background: ink(strandById.get(b.strand)?.color || '#888') }} /> {b.title || b.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="machine-zoom">
          {group ? (
            <span className="dag-legend">
              <span><i className="lg-line up" />Earlier steps</span>
              <span><i className="lg-line down" />Later steps</span>
              <span><i className="lg-line cross" />From / to another group</span>
            </span>
          ) : (
            <span className="dag-legend"><span>Arrows show which group's output another group uses</span></span>
          )}
        </div>
      </div>

      <div className="machine-body">
        <div ref={vpRef} className={`dag-viewport ${group ? 'gv-scroll' : ''}`}>
          {focusScene && (
            <div className="scene-focus-banner" style={{ '--focus-c': focusScene.color }}>
              <span className="scene-focus-dot" />
              <span>Slide {pad2(focusScene.num)} · <b>{focusScene.title}</b> ({focusScene.beads.length} steps)</span>
              {onClearFocus && <button type="button" className="link-btn" onClick={onClearFocus}>Clear</button>}
            </div>
          )}
          {group && groupLayout ? renderGroup() : renderMap()}
          <div className="gm-hint">
            {group
              ? 'Scroll down or press ↑ to go back to all groups · Double-click a function to see its code'
              : 'Scroll up on a group, double-click it, or press Open to see its functions'}
          </div>
        </div>
        {group ? beadInspector : groupInspector}
      </div>
    </div>
  );
}
