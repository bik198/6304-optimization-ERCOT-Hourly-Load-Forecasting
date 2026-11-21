/** One chapter: number and eyebrow on one line, a headline, then content. */
export default function Chapter({ id, no, eyebrow, title, children }) {
  return (
    <section id={id} className="chapter" aria-labelledby={`${id}-title`}>
      <span className="eyebrow">{no} · {eyebrow}</span>
      <h2 id={`${id}-title`}>{title}</h2>
      {children}
    </section>
  );
}
