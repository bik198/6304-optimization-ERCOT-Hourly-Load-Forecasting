/**
 * Real IEEE-754 double-precision (64-bit) float-to-binary converter for Depth 4.
 * Every 0 and 1 shown at the "silicon floor" comes directly from DataView.setFloat64().
 */
export function floatToBits(num) {
  const val = Number(num);
  if (!Number.isFinite(val)) {
    return {
      value: String(num),
      sign: '0',
      exponent: '11111111111',
      mantissa: '0'.repeat(52),
      full64: '0' + '11111111111' + '0'.repeat(52),
      hex: '0x7FF0000000000000',
    };
  }
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setFloat64(0, val, false); // big-endian IEEE-754 layout

  let bits = '';
  let hex = '0x';
  for (let i = 0; i < 8; i++) {
    const b = view.getUint8(i);
    bits += b.toString(2).padStart(8, '0');
    hex += b.toString(16).toUpperCase().padStart(2, '0');
  }

  return {
    value: val,
    sign: bits.slice(0, 1),
    exponent: bits.slice(1, 12),
    mantissa: bits.slice(12),
    full64: bits,
    hex,
  };
}

/** Match a query string against a bead's tokens, name, title, strand, io, or explanation. */
export function matchBead(bead, query) {
  const q = (query || '').trim().toLowerCase();
  if (!q) return { matched: true, hitTokens: [] };
  const hitTokens = (bead.tokens || []).filter(
    t => t.toLowerCase().includes(q) || q.includes(t.toLowerCase()),
  );
  const textHit =
    bead.name.toLowerCase().includes(q) ||
    (bead.title || '').toLowerCase().includes(q) ||
    bead.strand.toLowerCase().includes(q) ||
    bead.explains.toLowerCase().includes(q) ||
    (bead.io || []).some(s => s.toLowerCase().includes(q));
  return { matched: hitTokens.length > 0 || textHit, hitTokens };
}
