export const MAX_GENRES = 5

export default function GenrePicker({ genres, value, onChange }) {
  const full = value.length >= MAX_GENRES
  const toggle = (g) => onChange(value.includes(g) ? value.filter((x) => x !== g) : [...value, g])
  return (
    <fieldset className="field">
      <legend>Genres ({value.length}/{MAX_GENRES})</legend>
      <div className="chips">
        {genres.map((g) => {
          const on = value.includes(g)
          return (
            <button
              key={g}
              type="button"
              className={on ? 'toggle on' : 'toggle'}
              aria-pressed={on}
              disabled={!on && full}
              onClick={() => toggle(g)}
            >
              {g}
            </button>
          )
        })}
      </div>
    </fieldset>
  )
}
