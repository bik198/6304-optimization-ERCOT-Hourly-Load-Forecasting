// Light / Semi-light / Dark. The choice is stored per browser; with no stored choice the
// page follows the OS setting (see the inline script in index.html, which runs before paint).
import { useState } from 'react';
import { getTheme } from '../charts/kit';

const KEY = 'gridstory.theme';
const OPTIONS = [['light', 'Light'], ['semi', 'Semi'], ['dark', 'Dark']];

export default function ThemeSwitch() {
  const [theme, setTheme] = useState(getTheme);
  const choose = t => {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem(KEY, t); } catch { /* per-browser nicety only */ }
    setTheme(t);
  };
  return (
    <div className="theme-switch" role="group" aria-label="Theme">
      {OPTIONS.map(([id, label]) => (
        <button key={id} type="button" aria-pressed={theme === id} onClick={() => choose(id)}>{label}</button>
      ))}
    </div>
  );
}
