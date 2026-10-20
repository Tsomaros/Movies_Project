// The creator form's values, checks and request, shared by the form and the compare panel.
import { MAX_GENRES } from './components/GenrePicker.jsx'

export const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
  'October', 'November', 'December']
export const RUNTIME = [40, 300]
export const YEARS = [1980, 2040]
export const MAX_BUDGET_M = 2000 // the API takes up to $2 billion
export const MAX_OVERVIEW = 1000

export const initialForm = () => ({
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
export function validate(f) {
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

// Newcomers carry negative ids, which the API counts as people with no earlier films.
export function toRequest(f) {
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
export function snapshot(f, request, languages) {
  return {
    genres: request.genres,
    runtime: request.runtime,
    monthName: MONTHS[Number(f.month) - 1],
    year: Number(f.year),
    languageName: languageName(f.language, languages),
    director: f.director,
    actors: f.actors,
    company: f.company,
    budget: request.budget,
    overview: request.overview,
  }
}

export const languageName = (code, languages) => languages.find((l) => l.code === code)?.name ?? code

const FIELD_OF = { genres: 'genres', runtime: 'runtime', release_date: 'release_date', budget: 'budget',
  overview: 'overview' }

// An ApiError as {fields, message}: errors for known inputs by form key, the rest in one message.
export function splitErrors(err) {
  const fields = {}
  const other = []
  for (const { field, message } of err.fields ?? []) {
    const key = FIELD_OF[field.split('.')[0]]
    if (key) fields[key] = message
    else other.push(`${field || 'input'}: ${message}`)
  }
  return { fields, message: [err.message, ...other].join(' ') }
}
