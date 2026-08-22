/**
 * Dünne SVD über einseitige Jacobi-Rotationen — exakt und robust für die
 * Matrixgrößen der App (bis einige hundert Zeilen/Spalten). Liefert M = U·diag(s)·Vᵀ
 * mit orthonormalen Spalten von U und V, absteigend sortierten Singulärwerten.
 *
 * Für sehr große, dichte Matrizen greift später der randomisierte Worker-Pfad
 * (Spezifikation §8); für die CA-Ansicht der geladenen Daten ist Jacobi exakt.
 */
export interface SVD { U: Float64Array; s: Float64Array; V: Float64Array; m: number; n: number; k: number; }

/** M ist zeilen-major (m×n). */
export function svd(M: Float64Array, m: number, n: number, eps = 1e-12): SVD {
  // Einseitiges Jacobi bevorzugt „hohe" Matrizen (m ≥ n); sonst transponiert rechnen.
  if (m < n) {
    const T = transpose(M, m, n);
    const r = svd(T, n, m, eps);
    return { U: r.V, s: r.s, V: r.U, m, n, k: r.k };
  }
  const A = Float64Array.from(M);           // wird zu U·diag(s)
  const V = identity(n);
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) { const aip = A[i * n + p], aiq = A[i * n + q]; alpha += aip * aip; beta += aiq * aiq; gamma += aip * aiq; }
        if (alpha === 0 || beta === 0) continue;
        if (Math.abs(gamma) <= eps * Math.sqrt(alpha * beta)) continue;
        off += gamma * gamma;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let i = 0; i < m; i++) { const aip = A[i * n + p], aiq = A[i * n + q]; A[i * n + p] = c * aip - s * aiq; A[i * n + q] = s * aip + c * aiq; }
        for (let i = 0; i < n; i++) { const vip = V[i * n + p], viq = V[i * n + q]; V[i * n + p] = c * vip - s * viq; V[i * n + q] = s * vip + c * viq; }
      }
    }
    if (off < 1e-24) break;
  }
  // Singulärwerte = Spaltennormen von A; U = A / s
  const sv: Array<{ s: number; j: number }> = [];
  for (let j = 0; j < n; j++) { let nrm = 0; for (let i = 0; i < m; i++) nrm += A[i * n + j] * A[i * n + j]; sv.push({ s: Math.sqrt(nrm), j }); }
  sv.sort((a, b) => b.s - a.s);
  const k = n;
  const U = new Float64Array(m * k), s = new Float64Array(k), Vo = new Float64Array(n * k);
  for (let c = 0; c < k; c++) {
    const { s: sig, j } = sv[c]; s[c] = sig;
    if (sig > 1e-14) for (let i = 0; i < m; i++) U[i * k + c] = A[i * n + j] / sig;
    for (let i = 0; i < n; i++) Vo[i * k + c] = V[i * n + j];
  }
  return { U, s, V: Vo, m, n, k };
}

function transpose(M: Float64Array, m: number, n: number): Float64Array {
  const T = new Float64Array(n * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = M[i * n + j];
  return T;
}
function identity(n: number): Float64Array { const I = new Float64Array(n * n); for (let i = 0; i < n; i++) I[i * n + i] = 1; return I; }
