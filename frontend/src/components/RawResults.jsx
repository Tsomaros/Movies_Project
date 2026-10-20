// A plain view of the /predict response, until the results page replaces it.
const dollars = (x) =>
  x >= 1e9 ? `$${(x / 1e9).toFixed(2)}B` : x >= 1e6 ? `$${(x / 1e6).toFixed(1)}M` : `$${Math.round(x).toLocaleString()}`
const score = (x) => x.toFixed(1)

const KINDS = [
  { key: 'budget', title: 'Budget', fmt: dollars },
  { key: 'revenue', title: 'Worldwide revenue', fmt: dollars },
  { key: 'score', title: 'Audience score (0–10)', fmt: score },
]

export default function RawResults({ result }) {
  const prior = result.prior_films
  return (
    <section className="results" aria-live="polite">
      <h2>Prediction</h2>
      {result.warnings.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
      {KINDS.map(({ key, title, fmt }) => {
        const r = result[key]
        return (
          <div key={key} className="result-block">
            <h3>{title}{r.label ? <span className="tag">{r.label}</span> : null}</h3>
            <p>
              <strong>{fmt(r.low)} – {fmt(r.high)}</strong> (most likely {fmt(r.point)})
            </p>
            {r.given != null && <p className="hint">You entered a budget of {dollars(r.given)}.</p>}
            <ul>
              {r.drivers.map((d) => <li key={d.factor}>{d.text}</li>)}
            </ul>
          </div>
        )
      })}
      <p className="hint">
        Earlier films in our data: director {prior.director}, best-known lead actor {prior.actors_max},
        company {prior.company}.
      </p>
    </section>
  )
}
