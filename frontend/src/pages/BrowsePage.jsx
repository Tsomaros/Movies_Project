import { useEffect, useState } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router-dom'

import { get } from '../api.js'

const PAGE_SIZE = 24
const YEARS = [1900, 2100] // the API's limits
const SEARCH_DELAY = 300 // ms after typing stops
const FILTERS = ['q', 'genre', 'from', 'to']

const validYear = (s) => /^\d{4}$/.test(s) && Number(s) >= YEARS[0] && Number(s) <= YEARS[1]

function MovieCard({ movie, from }) {
  const votes = movie.vote_count ? ` (${movie.vote_count.toLocaleString('en-US')} votes)` : ''
  return (
    <li>
      <Link to={`/movies/${movie.id}`} state={{ from }} className="movie-card">
        {movie.poster_url
          ? <img src={movie.poster_url} alt="" loading="lazy" className="poster" />
          : <span className="poster poster-missing" aria-hidden="true">{movie.title}</span>}
        <span className="movie-title">{movie.title}</span>
        <span className="movie-meta">{[movie.release_year, movie.genres.slice(0, 2).join(', ')].filter(Boolean).join(' · ')}</span>
        {movie.vote_average != null && movie.vote_count > 0 && (
          <span className="movie-meta">★ {movie.vote_average.toFixed(1)}{votes}</span>
        )}
      </Link>
    </li>
  )
}

// The filters live in the URL, so reload, back/forward and shared links keep them.
export default function BrowsePage() {
  const [params, setParams] = useSearchParams()
  const location = useLocation()
  const [genres, setGenres] = useState([])
  const q = params.get('q') ?? ''
  const genre = params.get('genre') ?? ''
  const yearFrom = params.get('from') ?? ''
  const yearTo = params.get('to') ?? ''
  const page = Math.max(1, Number.parseInt(params.get('page'), 10) || 1)

  // The typed fields (title and years) reach the URL together, once typing stops.
  const [draft, setDraft] = useState({ q, from: yearFrom, to: yearTo })
  const [state, setState] = useState({ data: null, error: null, loading: true })

  // Sets filters and goes back to page 1; an empty value removes it from the URL.
  function setFilters(values, { replace = false } = {}) {
    setParams((p) => {
      const next = new URLSearchParams(p)
      for (const [key, value] of Object.entries(values)) {
        if (value) next.set(key, value)
        else next.delete(key)
      }
      next.delete('page')
      return next
    }, { replace })
  }

  useEffect(() => {
    get('/meta/genres').then(setGenres).catch(() => setGenres([]))
  }, [])

  // Back/forward changes the URL; keep the typed fields in step with it.
  useEffect(() => setDraft({ q, from: yearFrom, to: yearTo }), [q, yearFrom, yearTo])

  useEffect(() => {
    const typed = { q: draft.q.trim(), from: draft.from, to: draft.to }
    if (typed.q === q && typed.from === yearFrom && typed.to === yearTo) return
    const timer = setTimeout(() => setFilters(typed, { replace: true }), SEARCH_DELAY)
    return () => clearTimeout(timer)
  }, [draft]) // eslint-disable-line react-hooks/exhaustive-deps

  const type = (key) => (e) => setDraft((d) => ({ ...d, [key]: e.target.value }))

  useEffect(() => {
    const controller = new AbortController()
    setState((s) => ({ ...s, error: null, loading: true }))
    get('/movies', {
      q: q || undefined,
      genre: genre || undefined,
      year_from: validYear(yearFrom) ? yearFrom : undefined,
      year_to: validYear(yearTo) ? yearTo : undefined,
      page,
      page_size: PAGE_SIZE,
    }, controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e) => {
        if (e.name !== 'AbortError') setState({ data: null, error: e.message, loading: false })
      })
    return () => controller.abort()
  }, [q, genre, yearFrom, yearTo, page])

  function goTo(p) {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      if (p > 1) next.set('page', String(p))
      else next.delete('page')
      return next
    })
    window.scrollTo({ top: 0 })
  }

  const filtered = FILTERS.some((k) => params.get(k))
  const clear = () => { setDraft({ q: '', from: '', to: '' }); setParams({}) }
  const { data } = state
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1
  const yearError = [yearFrom, yearTo].some((y) => y && !validYear(y))

  return (
    <section>
      <h1>Browse movies</h1>
      <p className="muted">Search and filter the films in our data, the ones with the most votes first.</p>

      <form className="filters" role="search" onSubmit={(e) => e.preventDefault()}>
        <div className="field filter-search">
          <label htmlFor="browse-q">Title</label>
          <input id="browse-q" type="search" placeholder="Search titles" value={draft.q} maxLength={100}
            onChange={type('q')} />
        </div>
        <div className="field">
          <label htmlFor="browse-genre">Genre</label>
          <select id="browse-genre" value={genre} onChange={(e) => setFilters({ genre: e.target.value })}>
            <option value="">Any genre</option>
            {genres.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </div>
        <div className="field">
          <span className="label">Released</span>
          <div className="row">
            <input aria-label="From year" type="number" inputMode="numeric" placeholder="From" min={YEARS[0]}
              max={YEARS[1]} value={draft.from} onChange={type('from')} />
            <input aria-label="To year" type="number" inputMode="numeric" placeholder="To" min={YEARS[0]}
              max={YEARS[1]} value={draft.to} onChange={type('to')} />
          </div>
        </div>
        {filtered && <button type="button" className="link-button" onClick={clear}>Clear filters</button>}
      </form>
      {yearError && <p className="error">Years must be from {YEARS[0]} to {YEARS[1]}; that filter is ignored.</p>}

      {state.error && <p className="error" role="alert">{state.error}</p>}
      {data && (
        <div aria-busy={state.loading}>
          <p className="muted result-count" aria-live="polite">
            {state.loading ? 'Loading…' : `${data.total.toLocaleString('en-US')} movie${data.total === 1 ? '' : 's'}`}
          </p>
          {data.items.length === 0 && !state.loading ? (
            <p>
              No movies match these filters.{' '}
              <button type="button" className="link-button" onClick={clear}>Clear filters</button>
            </p>
          ) : (
            <ul className="movie-grid">
              {data.items.map((m) => <MovieCard key={m.id} movie={m} from={location.pathname + location.search} />)}
            </ul>
          )}
          {pages > 1 && (
            <nav className="pager" aria-label="Pages">
              <button type="button" disabled={page <= 1} onClick={() => goTo(page - 1)}>← Previous</button>
              <span>Page {page} of {pages.toLocaleString('en-US')}</span>
              <button type="button" disabled={page >= pages} onClick={() => goTo(page + 1)}>Next →</button>
            </nav>
          )}
        </div>
      )}
      {!data && state.loading && <p className="muted">Loading…</p>}
    </section>
  )
}
