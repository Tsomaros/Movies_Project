import { useCallback, useEffect, useRef, useState } from 'react'

import { get, post, searchCompanies, searchPeople } from '../api.js'
import ActorPicker from '../components/ActorPicker.jsx'
import GenrePicker from '../components/GenrePicker.jsx'
import Results from '../components/Results.jsx'
import SinglePicker from '../components/SinglePicker.jsx'
import { ErrorMessage, Loading } from '../components/Status.jsx'
import { initialForm, MAX_OVERVIEW, MONTHS, RUNTIME, snapshot, splitErrors, toRequest, validate, YEARS } from '../film.js'

// Defined once so the autocompletes don't restart their search on every render.
const searchDirectors = searchPeople('director')
const searchActors = searchPeople('actor')

let lastResultId = 0

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

  const loadMeta = useCallback(() => {
    setMeta((m) => ({ ...m, error: null, loading: true }))
    Promise.all([get('/meta/genres'), get('/meta/languages')])
      .then(([genres, langs]) =>
        setMeta({ genres, languages: langs.languages, note: langs.note, error: null, loading: false }))
      .catch((e) => setMeta((m) => ({ ...m, error: e.message, loading: false })))
  }, [])

  useEffect(loadMeta, [loadMeta])

  const set = (key) => (value) => setForm((f) => ({ ...f, [key]: value }))
  const onInput = (key) => (e) => set(key)(e.target.value)

  // Each result keeps the raw form it came from, so the compare panel changes one thing from that.
  function show(submitted, data) {
    const request = toRequest(submitted)
    setResult({ id: ++lastResultId, data, form: submitted, film: snapshot(submitted, request, meta.languages) })
  }

  async function run() {
    const found = validate(form)
    setErrors(found)
    setRequestError(null)
    if (Object.keys(found).length) return
    setSubmitting(true)
    try {
      const submitted = form
      show(submitted, await post('/predict', toRequest(submitted)))
    } catch (err) {
      const { fields, message } = splitErrors(err)
      setErrors(fields)
      setRequestError(message)
    } finally {
      setSubmitting(false)
    }
  }

  function keep(changed, data) {
    setForm(changed)
    setErrors({})
    setRequestError(null)
    show(changed, data)
  }

  if (meta.loading) return <Loading>Loading the form…</Loading>
  if (meta.error) return <ErrorMessage message={meta.error} onRetry={loadMeta} />

  const errorText = (name) => (errors[name] ? <p className="error">{errors[name]}</p> : null)

  return (
    <section>
      <h1>Predict a movie</h1>
      <p className="muted">
        Describe a planned film to get estimated ranges for its budget, box-office revenue and audience score.
      </p>

      <form className="predict-form" onSubmit={(e) => { e.preventDefault(); run() }} noValidate>
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
            Pick names from the list. Someone not in our data can be added as a newcomer with no earlier films.
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
          {submitting && <Loading>Running the models, this takes a second or two…</Loading>}
          {requestError && (
            <ErrorMessage message={requestError} onRetry={Object.keys(errors).length ? null : run} />
          )}
        </div>
      </form>

      {result && (
        <Results key={result.id} result={result.data} film={result.film} form={result.form} meta={meta}
          onKeep={keep} busy={submitting} headingRef={resultsHeading} />
      )}
    </section>
  )
}
