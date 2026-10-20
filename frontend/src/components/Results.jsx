import { direction, driverSentence } from '../explain.js'
import { dollars, ratio, score } from '../format.js'
import Compare from './Compare.jsx'
import Limitations from './Limitations.jsx'
import RangeChart from './RangeChart.jsx'

const ARROW = { up: '▲', down: '▼', flat: '•' }

function Drivers({ kind, drivers, film, prior }) {
  const largest = Math.max(...drivers.map((d) => Math.abs(d.effect)), 1e-9)
  return (
    <>
      <p className="drivers-title">Top factors, compared with a typical film in our data</p>
      <ul className="drivers">
        {drivers.map((d) => {
          const dir = direction(kind, d.effect)
          return (
            <li key={d.factor} className={`driver ${dir}`}>
              <span className="driver-arrow" aria-hidden="true">{ARROW[dir]}</span>
              <span className="driver-bar" aria-hidden="true">
                <span style={{ width: `${(Math.abs(d.effect) / largest) * 100}%` }} />
              </span>
              <span>{driverSentence(kind, d, film, prior)}</span>
            </li>
          )
        })}
      </ul>
    </>
  )
}

function Card({ kind, title, r, tag, headline, sub, notes = [], chart, film, prior }) {
  return (
    <article className="result-card">
      <h3>{title}{tag && <span className="tag">{tag}</span>}</h3>
      <p className="headline">{headline}</p>
      {sub && <p className="sub">{sub}</p>}
      {notes.map((n) => <p key={n} className="hint">{n}</p>)}
      <RangeChart low={r.low} point={r.point} high={r.high} {...chart} />
      <Drivers kind={kind} drivers={r.drivers} film={film} prior={prior} />
    </article>
  )
}

function budgetCard(r) {
  const range = `${dollars(r.low)} – ${dollars(r.high)}`
  const chart = { scale: 'log', fmt: dollars, label: `Budget range ${range}, most likely ${dollars(r.point)}` }
  if (r.given == null) {
    return { headline: range, sub: `Most likely ${dollars(r.point)}`, chart }
  }
  const notes = []
  if (r.given < r.low) notes.push('Your budget is below the usual range for films like this.')
  if (r.given > r.high) notes.push('Your budget is above the usual range for films like this.')
  return {
    headline: `Your budget: ${dollars(r.given)}`,
    sub: `Films like this usually cost ${range} (most likely ${dollars(r.point)}).`,
    notes,
    chart: { ...chart, label: `${chart.label}; your budget ${dollars(r.given)}`,
      markers: [{ value: r.given, label: 'Your budget' }] },
  }
}

function revenueCard(r, budget) {
  const range = `${dollars(r.low)} – ${dollars(r.high)}`
  const given = budget.given
  const ref = given ?? budget.point
  const refLabel = given != null ? 'Your budget' : 'Estimated budget'
  return {
    headline: range,
    sub: `Most likely ${dollars(r.point)}${given != null ? `, ${ratio(r.point / given)} your budget` : ''}`,
    notes: r.model === 'lgbm_no_budget'
      ? ['Estimated without a budget. Enter one to use the more accurate revenue model.'] : [],
    chart: { scale: 'log', fmt: dollars, label: `Revenue range ${range}, most likely ${dollars(r.point)}; `
      + `${refLabel.toLowerCase()} ${dollars(ref)}`, markers: [{ value: ref, label: refLabel }] },
  }
}

function scoreCard(r) {
  const range = `${score(r.low)} – ${score(r.high)}`
  return {
    headline: `${range} out of 10`,
    sub: `Most likely ${score(r.point)}`,
    chart: { scale: 'linear', fmt: (x) => String(x), label: `Score range ${range} out of 10, most likely ${score(r.point)}` },
  }
}

function PriorFilms({ film, prior }) {
  const count = (n) => (n ? String(n) : 'none, a newcomer')
  const parts = [
    film.director && `${film.director.name}: ${count(prior.director)}`,
    film.actors.length > 0 && `best-known lead actor: ${prior.actors_max || 'none, all newcomers'}`,
    film.company && `${film.company.name}: ${count(prior.company)}`,
  ].filter(Boolean)
  if (!parts.length) {
    return (
      <p className="hint">
        No director, actors or company were picked, so the prediction leans on the genres, runtime, release date
        and language.
      </p>
    )
  }
  const newcomers = [film.director, ...film.actors, film.company].filter((x) => x?.newcomer).map((x) => x.name)
  return (
    <p className="hint">
      Earlier films in our data, {parts.join(', ')}.
      {newcomers.length > 0 && ` Added as newcomers with no earlier films: ${newcomers.join(', ')}.`}
    </p>
  )
}

// The /predict response as ranges, charts and plain-English factors. `film` is the form as it was submitted
// for the wording, `form` the raw form for the compare panel; `busy` dims them while a new prediction runs.
export default function Results({ result, film, form, meta, onKeep, busy, headingRef }) {
  const prior = result.prior_films
  const cards = [
    { kind: 'budget', title: 'Budget', ...budgetCard(result.budget) },
    { kind: 'revenue', title: 'Worldwide box office', ...revenueCard(result.revenue, result.budget) },
    { kind: 'score', title: 'Audience score', ...scoreCard(result.score) },
  ]
  return (
    <section className={busy ? 'results busy' : 'results'} aria-busy={busy}>
      <h2 ref={headingRef} tabIndex={-1}>Prediction</h2>
      {result.warnings.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
      {cards.map((c) => (
        <Card key={c.kind} {...c} r={result[c.kind]} tag={result[c.kind].label} film={film} prior={prior} />
      ))}
      <PriorFilms film={film} prior={prior} />
      <Compare form={form} result={result} meta={meta} onKeep={onKeep} />
      <Limitations />
    </section>
  )
}
