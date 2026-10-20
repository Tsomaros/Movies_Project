import { useState } from 'react'

import { post } from '../api.js'
import { languageName, MONTHS, splitErrors, toRequest, validate } from '../film.js'
import { dollars, ratio, score } from '../format.js'
import GenrePicker from './GenrePicker.jsx'
import { ErrorMessage, Loading } from './Status.jsx'

// What the panel can change; each maps to one form key and the validate() error it can raise.
const FIELDS = {
  genres: { label: 'Genres', error: 'genres' },
  month: { label: 'Release month', error: 'release_date' },
  year: { label: 'Release year', error: 'release_date' },
  runtime: { label: 'Runtime', error: 'runtime' },
  language: { label: 'Original language', error: null },
  budgetM: { label: 'Budget', error: 'budget' },
}

function describe(key, value, languages) {
  switch (key) {
    case 'genres': return `the genre${value.length === 1 ? '' : 's'} ${value.join(', ')}`
    case 'month': return `a ${MONTHS[Number(value) - 1]} release`
    case 'year': return `releasing in ${value}`
    case 'runtime': return `a runtime of ${value} minutes`
    case 'language': return `${languageName(value, languages)} as the original language`
    case 'budgetM': return value.trim() === '' ? 'no budget given' : `a budget of ${dollars(Number(value) * 1e6)}`
    default: return String(value)
  }
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)

function Editor({ field, value, onChange, meta }) {
  const input = (props) => <input id="compare-value" value={value} onChange={(e) => onChange(e.target.value)} {...props} />
  switch (field) {
    case 'genres':
      return <GenrePicker genres={meta.genres} value={value} onChange={onChange} />
    case 'month':
      return (
        <select id="compare-value" value={value} onChange={(e) => onChange(e.target.value)}>
          {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
        </select>
      )
    case 'language':
      return (
        <select id="compare-value" value={value} onChange={(e) => onChange(e.target.value)}>
          {meta.languages.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
        </select>
      )
    case 'budgetM':
      return input({ type: 'number', inputMode: 'decimal', min: 0, step: 'any', placeholder: 'Empty for no budget' })
    default:
      return input({ type: 'number', inputMode: 'numeric' })
  }
}

// One result as "most likely" plus its range; a budget the creator gave is shown as is.
function Cell({ kind, r }) {
  if (kind === 'budget' && r.given != null) {
    return <><strong>{dollars(r.given)}</strong><span className="hint">your budget</span></>
  }
  const fmt = kind === 'score' ? score : dollars
  return <><strong>{fmt(r.point)}</strong><span className="hint">{fmt(r.low)} – {fmt(r.high)}</span></>
}

function difference(kind, a, b) {
  const before = kind === 'budget' ? a.given ?? a.point : a.point
  const after = kind === 'budget' ? b.given ?? b.point : b.point
  if (kind === 'score') {
    const d = Number(score(after)) - Number(score(before)) // as shown, so 6.7 → 6.6 isn't "no change"
    if (Math.abs(d) < 0.05) return { text: 'No change', dir: 'flat' }
    return { text: `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(1)} points`, dir: d > 0 ? 'up' : 'down' }
  }
  const x = after / before
  if (Math.abs(x - 1) < 0.01) return { text: 'No change', dir: 'flat' }
  if (x >= 2) return { text: `${ratio(x)} higher`, dir: 'up' }
  if (x <= 0.5) return { text: `${ratio(1 / x)} lower`, dir: 'down' }
  return { text: `${x > 1 ? '+' : '−'}${Math.round(Math.abs(x - 1) * 100)}%`, dir: x > 1 ? 'up' : 'down' }
}

const ROWS = [['budget', 'Budget'], ['revenue', 'Worldwide box office'], ['score', 'Audience score']]
const ARROW = { up: '▲', down: '▼', flat: '' }

function Table({ before, after, change }) {
  return (
    <div className="compare-table-wrap">
      <table className="compare-table">
        <caption>Most likely value and 10–90% range, as submitted and with {change}</caption>
        <thead>
          <tr><th scope="col" /><th scope="col">As submitted</th><th scope="col">With the change</th>
            <th scope="col">Difference</th></tr>
        </thead>
        <tbody>
          {ROWS.map(([kind, label]) => {
            const a = before[kind]
            const b = after[kind]
            const d = difference(kind, a, b)
            const given = kind === 'budget' && a.given != null && b.given != null
            const inside = d.dir !== 'flat' && !given && b.point >= a.low && b.point <= a.high
            return (
              <tr key={kind}>
                <th scope="row">{label}</th>
                <td><Cell kind={kind} r={a} /></td>
                <td><Cell kind={kind} r={b} /></td>
                <td>
                  <span className={`diff ${d.dir}`}>
                    {ARROW[d.dir] && <span aria-hidden="true">{ARROW[d.dir]} </span>}{d.text}
                  </span>
                  {inside && <span className="hint">still inside the first range</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// "Try changing one thing": a second prediction with one input changed, side by side with the first.
export default function Compare({ form, result, meta, onKeep }) {
  const [field, setField] = useState('month')
  const [value, setValue] = useState(form.month)
  const [state, setState] = useState({ loading: false, error: null, fieldError: null, compared: null })

  function choose(key) {
    setField(key)
    setValue(form[key])
    setState((s) => ({ ...s, error: null, fieldError: null }))
  }

  async function run() {
    const changed = { ...form, [field]: value }
    const fieldError = validate(changed)[FIELDS[field].error]
    if (fieldError) {
      setState((s) => ({ ...s, fieldError, error: null }))
      return
    }
    setState((s) => ({ ...s, loading: true, error: null, fieldError: null }))
    try {
      const data = await post('/predict', toRequest(changed))
      setState({ loading: false, error: null, fieldError: null,
        compared: { data, form: changed, change: describe(field, value, meta.languages) } })
    } catch (err) {
      const { fields, message } = splitErrors(err)
      setState((s) => ({ ...s, loading: false, fieldError: fields[FIELDS[field].error] ?? null,
        error: Object.keys(fields).length ? null : message }))
    }
  }

  const unchanged = same(value, form[field])
  const { compared } = state
  // Only what the change adds; the first prediction's warnings are already shown above.
  const newWarnings = compared ? compared.data.warnings.filter((w) => !result.warnings.includes(w)) : []

  return (
    <section className="compare" aria-labelledby="compare-title">
      <h3 id="compare-title">Try changing one thing</h3>
      <p className="hint">Change one input and see how the prediction moves; everything else stays as submitted.</p>
      <form className="compare-form" onSubmit={(e) => { e.preventDefault(); run() }} noValidate>
        <div className="field">
          <label htmlFor="compare-field">What to change</label>
          <select id="compare-field" value={field} onChange={(e) => choose(e.target.value)}>
            {Object.entries(FIELDS).map(([k, f]) => <option key={k} value={k}>{f.label}</option>)}
          </select>
        </div>
        <div className={field === 'genres' ? 'field compare-wide' : 'field'}>
          {field !== 'genres' && (
            <label htmlFor="compare-value">{field === 'budgetM' ? 'New budget (millions of US dollars)' : 'New value'}</label>
          )}
          <Editor field={field} value={value} onChange={setValue} meta={meta} />
          {state.fieldError && <p className="error">{state.fieldError}</p>}
        </div>
        <div className="compare-actions">
          <button type="submit" className="primary" disabled={state.loading || unchanged}>
            {state.loading ? 'Comparing…' : 'Compare'}
          </button>
          {unchanged && <p className="hint">Pick a value different from the one you submitted.</p>}
        </div>
      </form>

      {state.loading && <Loading>Predicting the changed version…</Loading>}
      {state.error && <ErrorMessage message={state.error} onRetry={run} />}
      {compared && (
        <div className={state.loading ? 'compare-result busy' : 'compare-result'} aria-live="polite">
          <Table before={result} after={compared.data} change={compared.change} />
          {newWarnings.length > 0 && <ul className="warnings">{newWarnings.map((w) => <li key={w}>{w}</li>)}</ul>}
          <button type="button" className="secondary" onClick={() => onKeep(compared.form, compared.data)}>
            Use this version
          </button>
          <p className="hint">This copies {compared.change} into the form and shows its full results above.</p>
        </div>
      )}
    </section>
  )
}
