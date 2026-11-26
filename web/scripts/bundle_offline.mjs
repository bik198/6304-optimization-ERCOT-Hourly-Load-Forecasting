// Bundles Vite's `dist/` output + JSON datasets + images + fonts into ONE self-contained
// USB-stick HTML file (`web/dist-offline/beadline-offline.html`) that runs from file:/// with zero server.
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const WEB_DIR = resolve(import.meta.dirname, '..');
const DIST_DIR = join(WEB_DIR, 'dist');
const DATA_DIR = join(WEB_DIR, 'public', 'data');
const OUT_DIR = join(WEB_DIR, 'dist-offline');
mkdirSync(OUT_DIR, { recursive: true });

const assetsDir = join(DIST_DIR, 'assets');
const assetFiles = readdirSync(assetsDir);

const cssFile = assetFiles.find(f => f.endsWith('.css'));
const jsFile = assetFiles.find(f => f.endsWith('.js'));
if (!cssFile || !jsFile) {
  throw new Error('Run `npm run build` before `bundle_offline.mjs`.');
}

// Inline woff2 font URLs inside CSS as base64 data URIs so zero external file requests occur on file:///
let cssText = readFileSync(join(assetsDir, cssFile), 'utf8');
cssText = cssText.replace(/url\(\s*['"]?(?:\.\/|\.\.\/assets\/|assets\/)?([^'")?#]+\.woff2)['"]?\s*\)/g, (m, fname) => {
  const fpath = join(assetsDir, fname);
  try {
    const b64 = readFileSync(fpath).toString('base64');
    return `url("data:font/woff2;base64,${b64}")`;
  } catch {
    return m;
  }
});

const jsText = readFileSync(join(assetsDir, jsFile), 'utf8');

const inlineData = {
  'beads.json': JSON.parse(readFileSync(join(DATA_DIR, 'beads.json'), 'utf8')),
  'story.json': JSON.parse(readFileSync(join(DATA_DIR, 'story.json'), 'utf8')),
  'eda.json': JSON.parse(readFileSync(join(DATA_DIR, 'eda.json'), 'utf8')),
  images: {},
};

const imgDir = join(DATA_DIR, 'img');
try {
  for (const img of readdirSync(imgDir)) {
    if (img.endsWith('.png') || img.endsWith('.jpg') || img.endsWith('.jpeg')) {
      const mime = img.endsWith('.png') ? 'image/png' : 'image/jpeg';
      inlineData.images[`img/${img}`] = `data:${mime};base64,${readFileSync(join(imgDir, img)).toString('base64')}`;
    }
  }
} catch {
  // optional images directory
}

const safeJson = JSON.stringify(inlineData).replace(/<\//g, '<\\/');
const safeJs = jsText.replace(/<\/script/gi, '<\\/script');
const docTitle = inlineData['story.json']?.project?.docTitle || 'Beadline Offline';

const html = `<!doctype html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="color-scheme" content="dark light" />
    <meta name="darkreader-lock" />
    <title>${docTitle} (Offline)</title>
    <style>${cssText}</style>
  </head>
  <body>
    <div id="root"></div>
    <script>window.__BEADLINE_INLINE_DATA__ = ${safeJson};</script>
    <script type="module">${safeJs}</script>
  </body>
</html>
`;

const outPath = join(OUT_DIR, 'beadline-offline.html');
writeFileSync(outPath, html, 'utf8');
const kb = (statSync(outPath).size / 1024).toFixed(1);
console.log(`wrote dist-offline/beadline-offline.html (${kb} KB standalone single-file build)`);
