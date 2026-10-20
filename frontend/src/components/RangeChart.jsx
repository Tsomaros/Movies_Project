// A horizontal range strip: the low–high band, a dot for the most likely value and optional markers.
// Dollar values use a log axis, since a range like $12M–$434M doesn't fit on a linear one.
const MAX_TICKS = 6
const PAD = 1.6 // log axis: room on each side of the outermost value
const FLOOR = 1e3 // log axis: values below this are drawn at it

function logTicks(lo, hi) {
  const ticks = []
  for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k++) {
    for (const m of [1, 2, 5]) {
      const t = m * 10 ** k
      if (t >= lo && t <= hi) ticks.push(t)
    }
  }
  if (ticks.length <= MAX_TICKS) return ticks
  const powers = ticks.filter((t) => Number.isInteger(Math.log10(t)))
  return powers.length >= 2 ? powers : ticks.filter((_, i) => i % 2 === 0)
}

function axis(scale, values) {
  if (scale === 'linear') {
    return { pos: (v) => v * 10, ticks: [0, 2, 4, 6, 8, 10] } // the 0–10 score
  }
  const vs = values.map((v) => Math.max(v, FLOOR))
  const lo = Math.min(...vs) / PAD
  const hi = Math.max(...vs) * PAD
  const span = Math.log(hi) - Math.log(lo)
  return { pos: (v) => ((Math.log(Math.max(v, FLOOR)) - Math.log(lo)) / span) * 100, ticks: logTicks(lo, hi) }
}

// Labels near an edge are anchored to that edge so they stay inside the chart.
const anchor = (left) => (left < 8 ? 'start' : left > 92 ? 'end' : 'middle')

export default function RangeChart({ low, point, high, scale, fmt, markers = [], label }) {
  const { pos, ticks } = axis(scale, [low, point, high, ...markers.map((m) => m.value)])
  const left = pos(low)
  return (
    <div className={`range-chart${markers.length ? ' has-markers' : ''}`} role="img" aria-label={label}>
      <div className="rc-track">
        {ticks.map((t) => <span key={t} className="rc-grid" style={{ left: `${pos(t)}%` }} />)}
        <span className="rc-band" style={{ left: `${left}%`, width: `${pos(high) - left}%` }}
          title={`Range: ${fmt(low)} – ${fmt(high)}`} />
        {markers.map((m) => (
          <span key={m.label} className="rc-marker" style={{ left: `${pos(m.value)}%` }} title={`${m.label}: ${fmt(m.value)}`}>
            <span className={`rc-marker-label ${anchor(pos(m.value))}`}>{m.label}</span>
          </span>
        ))}
        <span className="rc-point" style={{ left: `${pos(point)}%` }} title={`Most likely: ${fmt(point)}`} />
      </div>
      <div className="rc-axis" aria-hidden="true">
        {ticks.map((t) => (
          <span key={t} className={`rc-tick ${anchor(pos(t))}`} style={{ left: `${pos(t)}%` }}>{fmt(t)}</span>
        ))}
      </div>
    </div>
  )
}
