import { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'

import { get } from '../api.js'
import { dollars } from '../format.js'

const languageNames = new Intl.DisplayNames(['en'], { type: 'language' })

function languageName(code) {
  if (!code) return null
  try {
    return languageNames.of(code)
  } catch {
    return code
  }
}

function runtime(minutes) {
  if (!minutes) return null
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  return h ? `${h}h${m ? ` ${m}m` : ''}` : `${m}m`
}

function releaseDate(iso) {
  if (!iso) return null
  const d = new Date(`${iso}T00:00:00`)
  return Number.isNaN(d.getTime()) ? iso
    : d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
}

// Directors first, then the other key jobs in the order they first appear.
function crewByJob(movie) {
  const groups = new Map()
  if (movie.directors.length) groups.set(movie.directors.length > 1 ? 'Directors' : 'Director', movie.directors)
  for (const c of movie.crew) {
    if (!groups.has(c.job)) groups.set(c.job, [])
    groups.get(c.job).push(c)
  }
  return [...groups]
}

const names = (people) => people.map((p) => p.name).join(', ')

function Facts({ movie }) {
  const score = movie.vote_count
    ? `${movie.vote_average.toFixed(1)} / 10 from ${movie.vote_count.toLocaleString('en-US')} votes` : null
  const facts = [
    ['Released', releaseDate(movie.release_date)],
    ['Original language', languageName(movie.original_language)],
    ['Budget', movie.budget ? dollars(movie.budget) : 'Not reported'],
    ['Box office', movie.revenue ? dollars(movie.revenue) : 'Not reported'],
    ['Audience score', score ?? 'No votes yet'],
    ...crewByJob(movie).map(([job, people]) => [job, names(people)]),
    [movie.companies.length > 1 ? 'Companies' : 'Company', names(movie.companies) || null],
  ].filter(([, v]) => v)
  return (
    <dl className="facts">
      {facts.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export default function MoviePage() {
  const { id } = useParams()
  const location = useLocation()
  const back = location.state?.from ?? '/movies'
  const [state, setState] = useState({ movie: null, error: null, status: null })

  useEffect(() => {
    const controller = new AbortController()
    setState({ movie: null, error: null, status: null })
    get(`/movies/${encodeURIComponent(id)}`, {}, controller.signal)
      .then((movie) => setState({ movie, error: null, status: 200 }))
      .catch((e) => {
        if (e.name !== 'AbortError') setState({ movie: null, error: e.message, status: e.status })
      })
    return () => controller.abort()
  }, [id])

  useEffect(() => {
    if (state.movie) document.title = `${state.movie.title} · Movie Prediction`
    return () => { document.title = 'Movie Prediction' }
  }, [state.movie])

  const backLink = <Link to={back} className="back-link">← Back to movies</Link>
  const { movie } = state

  if (state.status === 404 || state.status === 422) {
    return (
      <section>
        {backLink}
        <h1>Movie not found</h1>
        <p className="muted">There's no movie with this id in our data.</p>
      </section>
    )
  }
  if (state.error) return <section>{backLink}<p className="error" role="alert">{state.error}</p></section>
  if (!movie) return <section>{backLink}<p className="muted">Loading…</p></section>

  const subtitle = [movie.release_year, movie.certification, runtime(movie.runtime), movie.genres.join(', ')]
    .filter(Boolean).join(' · ')
  return (
    <article className="movie-detail">
      {backLink}
      <div className="detail-layout">
        {movie.poster_url
          ? <img src={movie.poster_url} alt={`Poster of ${movie.title}`} className="poster detail-poster" />
          : <span className="poster poster-missing detail-poster" aria-hidden="true">{movie.title}</span>}
        <div className="detail-body">
          <h1>{movie.title}</h1>
          {movie.original_title && movie.original_title !== movie.title && (
            <p className="muted">Original title: {movie.original_title}</p>
          )}
          {subtitle && <p className="muted">{subtitle}</p>}
          {movie.overview && <p className="overview">{movie.overview}</p>}
          <Facts movie={movie} />
        </div>
      </div>

      {movie.cast.length > 0 && (
        <section className="cast">
          <h2>Cast</h2>
          <ul>
            {movie.cast.map((c, i) => (
              <li key={`${c.id}-${i}`}>
                <span className="cast-name">{c.name}</span>
                {c.character && <span className="muted"> as {c.character}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}
      <p className="hint">Budget and box office are in US dollars of the release year, as reported to TMDB.</p>
    </article>
  )
}
