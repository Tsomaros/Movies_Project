import { useEffect, useRef, useState } from 'react'

import { get, post, searchCompanies, searchPeople } from '../api.js'
import ActorPicker from '../components/ActorPicker.jsx'
import GenrePicker, { MAX_GENRES } from '../components/GenrePicker.jsx'
import Results from '../components/Results.jsx'
import SinglePicker from '../components/SinglePicker.jsx'

// Defined once so the autocompletes don't restart their search on every render.
const searchDirectors = searchPeople('director')
const searchActors = searchPeople('actor')

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
  'November', 'December']
const RUNTIME = [40, 300]
const YEARS = [1980, 2040]
const MAX_BUDGET_M = 2000 // the API takes up to $2 billion
const MAX_OVERVIEW = 1000

const initialForm = () => ({
  genres: [],
  runtime: '110',
  month: '6',
  year: String(new Date().getFullYear() + 1),
  language: 'en',
  director: null,
  actors: [],
  company: null,
  budgetM: '',
  overview: '',
})

const between = (x, [lo, hi]) => Number.isFinite(x) && x >= lo && x <= hi

// Client checks that mirror the API's schema; the server still has the last word.
function validate(f) {
  const errors = {}
  if (f.genres.length < 1 || f.genres.length > MAX_GENRES) errors.genres = `Pick 1 to ${MAX_GENRES} genres.`
  if (!Number.isInteger(Number(f.runtime)) || !between(Number(f.runtime), RUNTIME))
    errors.runtime = `Runtime must be a whole number of minutes from ${RUNTIME[0]} to ${RUNTIME[1]}.`
  if (!Number.isInteger(Number(f.year)) || !between(Number(f.year), YEARS))
    errors.release_date = `Year must be from ${YEARS[0]} to ${YEARS[1]}.`
  if (f.budgetM.trim() !== '') {
    const b = Number(f.budgetM)
    if (!(b > 0) || b > MAX_BUDGET_M) errors.budget = `Budget must be more than 0 and at most ${MAX_BUDGET_M} million.`
  }
  if (f.overview.length > MAX_OVERVIEW) errors.overview = `Overview must be at most ${MAX_OVERVIEW} characters.`
  return errors
}

function toRequest(f) {
  return {
    genres: f.genres,
    runtime: Number(f.runtime),
    release_date: `${f.year}-${String(f.month).padStart(2, '0')}-01`,
    original_language: f.language,
    director_id: f.director?.id ?? null,
    actor_ids: f.actors.map((a) => a.id),
    company_id: f.company?.id ?? null,
    budget: f.budgetM.trim() === '' ? null : Math.round(Number(f.budgetM) * 1e6),
    overview: f.overview.trim(),
  }
}

// The form as submitted, for the results' wording; later edits to the form don't change shown results.
function snapshot(f, request, languages) {
  return {
    genres: request.genres,
    runtime: request.runtime,
    monthName: MONTHS[Number(f.month) - 1],
    year: Number(f.year),
    languageName: languages.find((l) => l.code === f.language)?.name ?? f.language,
    director: f.director,
    actors: f.actors,
    company: f.company,
    budget: request.budget,
    overview: request.overview,
  }
}

const FIELD_OF = { genres: 'genres', runtime: 'runtime', release_date: 'release_date', budget: 'budget',
  overview: 'overview' }

export default function PredictPage() {
  const [meta, setMeta] = useState({ genres: [], languages: [], note: '', error: null, loading: true })
  const [form, setForm] = useState(initialForm)
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [requestError, setRequestError] = useState(null)
  const resultsHeading = useRef(null)

  useEffect(() => {
    const heading = resultsHeading.current
    if (!result || !heading) return
    heading.focus({ preventScroll: true })
    heading.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [result])

  useEffect(() => {
    Promise.all([get('/meta/genres'), get('/meta/languages')])
      .then(([genres, langs]) =>
        setMeta({ genres, languages: langs.languages, note: langs.note, error: null, loading: false }))
      .catch((e) => setMeta((m) => ({ ...m, error: e.message, loading: false })))
  }, [])

  const set = (key) => (value) => setForm((f) => ({ ...f, [key]: value }))
  const onInput = (key) => (e) => set(key)(e.target.value)

  async function onSubmit(e) {
    e.preventDefault()
    const found = validate(form)
    setErrors(found)
    setRequestError(null)
    if (Object.keys(found).length) return
    setSubmitting(true)
    try {
      const request = toRequest(form)
      const data = await post('/predict', request)
      setResult({ data, film: snapshot(form, request, meta.languages) })
    } catch (err) {
      const fieldErrors = {}
      const other = []
      for (const { field, message } of err.fields ?? []) {
        const key = FIELD_OF[field.split('.')[0]]
        if (key) fieldErrors[key] = message
        else other.push(`${field || 'input'}: ${message}`)
      }
      setErrors(fieldErrors)
      setRequestError([err.message, ...other].join(' '))
      setResult(null)
    } finally {
      setSubmitting(false)
    }
  }

  if (meta.loading) return <p className="muted">Loading the form…</p>
  if (meta.error) return <p className="error" role="alert">{meta.error}</p>

  const errorText = (name) => (errors[name] ? <p className="error">{errors[name]}</p> : null)

  return (
    <section>
      <h1>Predict a movie</h1>
      <p className="muted">
        Describe a planned film to get estimated ranges for its budget, box-office revenue and audience score.
      </p>

      <form className="predict-form" onSubmit={onSubmit} noValidate>
        <div className="wide">
          <GenrePicker genres={meta.genres} value={form.genres} onChange={set('genres')} />
          {errorText('genres')}
        </div>

        <div className="field">
          <label htmlFor="runtime">Runtime (minutes)</label>
          <input id="runtime" type="number" inputMode="numeric" min={RUNTIME[0]} max={RUNTIME[1]}
            value={form.runtime} onChange={onInput('runtime')} />
          {errorText('runtime')}
        </div>

        <div className="field">
          <span className="label">Release</span>
          <div className="row">
            <select aria-label="Release month" value={form.month} onChange={onInput('month')}>
              {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
            </select>
            <input aria-label="Release year" type="number" inputMode="numeric" min={YEARS[0]} max={YEARS[1]}
              value={form.year} onChange={onInput('year')} />
          </div>
          {errorText('release_date')}
        </div>

        <div className="field">
          <label htmlFor="language">Original language</label>
          <select id="language" value={form.language} onChange={onInput('language')}>
            {meta.languages.map((l) => <option key={l.code} value={l.code}>{l.name}</option>)}
          </select>
          <p className="hint">{meta.note}</p>
        </div>

        <div className="field">
          <label htmlFor="budget">Budget (optional, millions of US dollars today)</label>
          <input id="budget" type="number" inputMode="decimal" min="0" step="any" placeholder="e.g. 40"
            value={form.budgetM} onChange={onInput('budgetM')} />
          <p className="hint">Leave it empty to get a budget estimate instead.</p>
          {errorText('budget')}
        </div>

        <SinglePicker label="Director (optional)" placeholder="Search directors" search={searchDirectors}
          value={form.director} onChange={set('director')} />
        <SinglePicker label="Lead production company (optional)" placeholder="Search companies"
          search={searchCompanies} value={form.company} onChange={set('company')} />
        <div className="wide">
          <ActorPicker search={searchActors} value={form.actors} onChange={set('actors')} />
          <p className="hint">
            Only names picked from the list are used. Anyone not in our data counts as a newcomer.
          </p>
        </div>

        <div className="field wide">
          <label htmlFor="overview">Overview (optional)</label>
          <textarea id="overview" rows={4} maxLength={MAX_OVERVIEW} value={form.overview}
            placeholder="A short plot summary" onChange={onInput('overview')} />
          <p className="hint">{form.overview.length}/{MAX_OVERVIEW} characters</p>
          {errorText('overview')}
        </div>

        <div className="wide actions">
          <button type="submit" className="primary" disabled={submitting}>
            {submitting ? 'Predicting…' : 'Predict'}
          </button>
          {requestError && <p className="error" role="alert">{requestError}</p>}
        </div>
      </form>

      {result && <Results result={result.data} film={result.film} headingRef={resultsHeading} />}
    </section>
  )
}
