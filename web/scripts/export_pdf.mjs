// Exports the 8-scene Print Deck (?print=1) to a landscape PDF fallback (`web/dist-offline/ercot-8-scenes.pdf`).
import { createRequire } from 'node:module';
import { mkdirSync, statSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { join, resolve } from 'node:path';

const require = createRequire('c:/roshanUTA/judgeTest/package.json');
const { chromium } = require('playwright');

const WEB_DIR = resolve(import.meta.dirname, '..');
const OUT_DIR = join(WEB_DIR, 'dist-offline');
mkdirSync(OUT_DIR, { recursive: true });

async function isReachable(url) {
  try {
    const res = await fetch(url);
    return res.ok;
  } catch {
    return false;
  }
}

async function main() {
  const port = 4176;
  const targetUrl = `http://localhost:${port}/?print=1`;
  const proc = spawn('npx', ['vite', 'preview', '--port', String(port), '--strictPort'], {
    cwd: WEB_DIR,
    shell: true,
    stdio: 'ignore',
  });

  for (let i = 0; i < 30; i++) {
    if (await isReachable(`http://localhost:${port}/`)) break;
    await new Promise(r => setTimeout(r, 250));
  }

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    await page.goto(targetUrl, { waitUntil: 'networkidle' });
    await page.waitForSelector('.print-deck');
    await page.waitForTimeout(300);

    const outPdf = join(OUT_DIR, 'ercot-8-scenes.pdf');
    await page.pdf({
      path: outPdf,
      landscape: true,
      printBackground: true,
      preferCSSPageSize: true,
    });
    const kb = (statSync(outPdf).size / 1024).toFixed(1);
    console.log(`wrote dist-offline/ercot-8-scenes.pdf (${kb} KB landscape PDF fallback)`);
  } finally {
    await browser.close();
    proc.kill();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
