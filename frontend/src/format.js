// Number formats shared by the results.
export const dollars = (x) =>
  x >= 1e9 ? `$${trim(x / 1e9)}B` : x >= 1e6 ? `$${trim(x / 1e6)}M` : x >= 1e3 ? `$${Math.round(x / 1e3)}K`
    : `$${Math.round(x)}`

// Three significant digits at most: 1.25, 12.5, 125.
function trim(x) {
  return x >= 100 ? String(Math.round(x)) : x >= 10 ? x.toFixed(1).replace(/\.0$/, '') : x.toFixed(2).replace(/\.?0+$/, '')
}

export const score = (x) => x.toFixed(1)

export const ratio = (x) => `${x >= 10 ? Math.round(x) : x.toFixed(1)}×`
