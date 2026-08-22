/**
 * Zell-Farbmodell der Matrix.
 *
 * Reine Nachbildung der Shader-Formel `mix(uBg, base, 0.25 + 0.75*clamp(v,0,1))`
 * für den Canvas-2D-Fallback. So ist per Test garantiert, dass der 2D-Pfad exakt
 * dieselben Farben liefert wie der WebGL-Pfad.
 *
 * @param bg   Hintergrund/Leerzellen-Farbe als 0..1-RGB (wie das Shader-Uniform uBg)
 * @param col  Materialfarbe der Spalte als 0..255-RGB
 * @param v    normierte Häufigkeit 0..1
 * @returns    0..255-RGB (ganzzahlig)
 */
export function mixCell(
  bg: readonly [number, number, number],
  col: readonly [number, number, number],
  v: number,
): [number, number, number] {
  const t = 0.25 + 0.75 * (v < 0 ? 0 : v > 1 ? 1 : v), it = 1 - t;
  return [
    (bg[0] * 255 * it + col[0] * t) | 0,
    (bg[1] * 255 * it + col[1] * t) | 0,
    (bg[2] * 255 * it + col[2] * t) | 0,
  ];
}
