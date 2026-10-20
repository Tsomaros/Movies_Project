// Plain-English sentences for the /predict drivers. Each driver is a SHAP effect against a typical film
// in the data: on the log scale for budget and revenue, in score points for the score.
import { dollars, ratio } from './format.js'

const NOUN = { budget: 'the budget estimate', revenue: 'revenue', score: 'the score' }
const MIN_PCT = 0.01 // smaller effects "barely change" the prediction
const MIN_POINTS = 0.05

export function direction(kind, effect) {
  const size = kind === 'score' ? Math.abs(effect) : Math.abs(Math.expm1(effect))
  if (size < (kind === 'score' ? MIN_POINTS : MIN_PCT)) return 'flat'
  return effect > 0 ? 'up' : 'down'
}

function change(kind, effect) {
  const noun = NOUN[kind]
  const dir = direction(kind, effect)
  if (dir === 'flat') return `barely changes ${noun}`
  if (kind === 'score') {
    const points = Math.abs(effect).toFixed(1)
    return `${dir === 'up' ? 'raises' : 'lowers'} ${noun} by ${points} points`
  }
  const pct = Math.expm1(effect)
  if (pct >= 1) return `pushes ${noun} up about ${ratio(1 + pct)}`
  return dir === 'up' ? `pushes ${noun} up by about ${Math.round(pct * 100)}%`
    : `pulls ${noun} down by about ${Math.round(-pct * 100)}%`
}

const possessive = (name) => (name.endsWith('s') ? `${name}'` : `${name}'s`)
const films = (n) => `${n} earlier film${n === 1 ? '' : 's'}`

// `film` is the form as it was submitted; `prior` is the response's prior_films.
// Every subject is singular, so it always takes "pushes", "raises" and so on.
function subject(factor, film, prior) {
  switch (factor) {
    case "Director's track record":
      if (!film.director) return 'Having no known director'
      if (!prior.director) return `${film.director.name} having no earlier films in our data`
      return `${possessive(film.director.name)} track record (${films(prior.director)})`
    case "Lead actors' track record":
      if (!film.actors.length) return 'Having no known lead actors'
      if (!prior.actors_max) return 'The lead actors having no earlier films in our data'
      return film.actors.length === 1 ? `${possessive(film.actors[0].name)} track record` : "The lead actors' track record"
    case 'Production company':
      if (!film.company) return 'Having no known production company'
      if (!prior.company) return `${film.company.name} having no earlier films in our data`
      return `${possessive(film.company.name)} track record (${films(prior.company)})`
    case 'Budget':
      return film.budget != null ? `Your budget of ${dollars(film.budget)}` : 'The budget'
    case 'Runtime':
      return `A runtime of ${film.runtime} minutes`
    case 'Release year':
      return `Releasing in ${film.year}`
    case 'Release month':
      return `A ${film.monthName} release`
    case 'Genres':
      return film.genres.length === 1 ? `The genre (${film.genres[0]})` : `The genre mix (${film.genres.join(', ')})`
    case 'Language':
      return `The original language (${film.languageName})`
    case 'Overview':
      return film.overview ? 'The plot overview' : 'Leaving the overview blank'
    default:
      return null
  }
}

export function driverSentence(kind, driver, film, prior) {
  const who = subject(driver.factor, film, prior)
  return who ? `${who} ${change(kind, driver.effect)}.` : driver.text
}
