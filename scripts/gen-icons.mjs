// PWA-Icon-Generator — rastert das CombiTab-Gitterlogo als PNG.
// Dependency-frei: nur Node-Builtins (zlib). Aufruf: node scripts/gen-icons.mjs
import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "public");
const BUILD_OUT = join(ROOT, "build");
mkdirSync(OUT, { recursive: true });

// ── CRC32 ──
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(buf) { let c = 0xffffffff; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td), 0);
  return Buffer.concat([len, td, crc]);
}

function encodePNG(w, h, rgba) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA
  const raw = Buffer.alloc(h * (1 + w * 4));
  for (let y = 0; y < h; y++) { raw[y * (1 + w * 4)] = 0; rgba.copy(raw, y * (1 + w * 4) + 1, y * w * 4, (y + 1) * w * 4); }
  const idat = deflateSync(raw, { level: 9 });
  return Buffer.concat([sig, chunk("IHDR", ihdr), chunk("IDAT", idat), chunk("IEND", Buffer.alloc(0))]);
}

// ── Palette (aus theme.css / Shell-Logo) ──
const ACCENT = [210, 38, 48], BGL = [246, 244, 242], LIGHT = [226, 223, 218];
const mix = (a, b, t) => a.map((v, i) => Math.round(v * (1 - t) + b[i] * t));
const RED50 = mix(ACCENT, BGL, 0.5);
// 3×3-Muster wie im Shell-Logo
const GRID = [[ACCENT, RED50, LIGHT], [RED50, ACCENT, RED50], [LIGHT, RED50, ACCENT]];

function drawIcon(size, { maskable = false } = {}) {
  const px = Buffer.alloc(size * size * 4);
  const bg = maskable ? BGL : BGL;
  // Hintergrund füllen
  for (let i = 0; i < size * size; i++) { px[i * 4] = bg[0]; px[i * 4 + 1] = bg[1]; px[i * 4 + 2] = bg[2]; px[i * 4 + 3] = 255; }
  // Gitterfläche zentriert; maskable mit größerem Sicherheitsrand
  const pad = Math.round(size * (maskable ? 0.20 : 0.14));
  const area = size - 2 * pad;
  const gap = area * 0.06;
  const cell = (area - 2 * gap) / 3;
  const radius = cell * 0.16;
  for (let gy = 0; gy < 3; gy++) for (let gx = 0; gx < 3; gx++) {
    const col = GRID[gy][gx];
    const x0 = pad + gx * (cell + gap), y0 = pad + gy * (cell + gap);
    for (let y = Math.floor(y0); y < Math.ceil(y0 + cell); y++) {
      for (let x = Math.floor(x0); x < Math.ceil(x0 + cell); x++) {
        if (x < 0 || y < 0 || x >= size || y >= size) continue;
        // abgerundete Ecken
        const lx = x - x0, ly = y - y0;
        const cx = Math.min(lx, cell - 1 - lx), cy = Math.min(ly, cell - 1 - ly);
        if (cx < radius && cy < radius) { const dx = radius - cx, dy = radius - cy; if (dx * dx + dy * dy > radius * radius) continue; }
        const i = (y * size + x) * 4;
        px[i] = col[0]; px[i + 1] = col[1]; px[i + 2] = col[2]; px[i + 3] = 255;
      }
    }
  }
  return encodePNG(size, size, px);
}

writeFileSync(join(OUT, "icon-192.png"), drawIcon(192));
writeFileSync(join(OUT, "icon-512.png"), drawIcon(512));
writeFileSync(join(OUT, "icon-maskable-512.png"), drawIcon(512, { maskable: true }));

// Quellbild für die Desktop-Pakete. electron-builder leitet daraus selbst das
// Windows-.ico, das macOS-.icns und die Linux-Icongrößen ab — deshalb genügt
// eine einzige, ausreichend große PNG-Datei und es braucht kein ImageMagick
// auf dem Build-Rechner.
mkdirSync(BUILD_OUT, { recursive: true });
writeFileSync(join(BUILD_OUT, "icon.png"), drawIcon(1024));

// Icon-Satz für Linux. electron-builder legt jede Größe unter
// /usr/share/icons/hicolor/<größe>/apps/ ab. Ein einzelnes 1024er-Icon reicht
// nicht: Panels und Anwendungsmenüs suchen die gängigen Stufen und zeigen sonst
// ein Platzhaltersymbol.
const LINUX_SIZES = [16, 24, 32, 48, 64, 128, 256, 512];
const ICONS_OUT = join(BUILD_OUT, "icons");
mkdirSync(ICONS_OUT, { recursive: true });
for (const size of LINUX_SIZES) {
  writeFileSync(join(ICONS_OUT, `${size}x${size}.png`), drawIcon(size));
}

// Vektor-Favicon (scharf in jeder Größe)
const rgbHex = (c) => "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
const cells = GRID.flatMap((row, gy) => row.map((c, gx) => `<rect x="${1 + gx * 13}" y="${1 + gy * 13}" width="11" height="11" rx="2" fill="${rgbHex(c)}"/>`)).join("");
writeFileSync(join(OUT, "favicon.svg"), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40" rx="6" fill="${rgbHex(BGL)}"/>${cells}</svg>\n`);

console.log("PWA-Icons in public/: icon-192.png, icon-512.png, icon-maskable-512.png, favicon.svg");
console.log("Paket-Icon in build/: icon.png (1024×1024)");
console.log(`Linux-Icon-Satz in build/icons/: ${LINUX_SIZES.map((s) => `${s}px`).join(", ")}`);
