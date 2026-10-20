import { useEffect, useId, useState } from 'react'

const MIN_CHARS = 2
const DELAY_MS = 250

// TMDB ids are positive, so a negative id never matches anyone in our data: the API counts it as a newcomer.
let lastNewcomerId = 0
const newcomer = (name) => ({ id: --lastNewcomerId, name, newcomer: true })

// A search box over one of the API's name searches. Only a name picked from the list is passed to onSelect;
// with `allowNewcomer`, the last option adds the typed name as someone with no earlier films in our data.
export default function Autocomplete({ label, placeholder, search, onSelect, exclude = [], disabled = false,
  allowNewcomer = false }) {
  const id = useId()
  const [text, setText] = useState('')
  const [options, setOptions] = useState([])
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)

  useEffect(() => {
    const q = text.trim()
    if (q.length < MIN_CHARS) {
      setOptions([])
      setStatus('idle')
      return
    }
    const controller = new AbortController()
    const timer = setTimeout(() => {
      setStatus('loading')
      search(q, controller.signal)
        .then((found) => {
          setOptions(found)
          setActive(-1)
          setStatus('done')
        })
        .catch((e) => {
          if (e.name !== 'AbortError') setStatus('error')
        })
    }, DELAY_MS)
    return () => {
      clearTimeout(timer)
      controller.abort() // a newer search replaces this one, so its answer must not land
    }
  }, [text, search])

  const typed = text.trim()
  const shown = options.filter((o) => !exclude.includes(o.id))
  const offerNewcomer = allowNewcomer && status === 'done' && typed.length >= MIN_CHARS
  const items = offerNewcomer ? [...shown, null] : shown // null stands for the newcomer option

  function pick(option) {
    onSelect(option ?? newcomer(typed))
    setText('')
    setOptions([])
    setOpen(false)
  }

  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, items.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && open && active >= 0 && active < items.length) {
      e.preventDefault()
      pick(items[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const listId = `${id}-list`
  const hintId = `${id}-hint`
  const showList = open && typed.length >= MIN_CHARS && status !== 'idle'
  const leftOver = !open && typed.length > 0 && !disabled

  return (
    <div className="autocomplete">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="text"
        value={text}
        placeholder={placeholder}
        disabled={disabled}
        autoComplete="off"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
        aria-describedby={leftOver ? hintId : undefined}
        onChange={(e) => {
          setText(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      {showList && (
        <ul id={listId} role="listbox" className="options">
          {status === 'loading' && shown.length === 0 && <li className="option-note">Searching…</li>}
          {status === 'error' && <li className="option-note">Search failed. Is the API running?</li>}
          {status === 'done' && shown.length === 0 && <li className="option-note">No matches in our data</li>}
          {items.map((o, i) => (
            <li
              key={o?.id ?? 'newcomer'}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={`option${o ? '' : ' option-new'}${i === active ? ' active' : ''}`}
              onMouseDown={(e) => e.preventDefault()} // keep focus so the click lands before blur closes the list
              onClick={() => pick(o)}
            >
              {o ? (
                <>
                  <span className="option-name">{o.name}</span>
                  <span className="option-meta">
                    {o.n_films} {o.n_films === 1 ? 'film' : 'films'}{o.known_for ? ` · ${o.known_for}` : ''}
                  </span>
                </>
              ) : (
                <>
                  <span className="option-name">Add “{typed}” as a newcomer</span>
                  <span className="option-meta">No earlier films in our data</span>
                </>
              )}
            </li>
          ))}
        </ul>
      )}
      {leftOver && (
        <p id={hintId} className="hint">
          {allowNewcomer ? 'Pick a name from the list or add it as a newcomer. ' : 'Pick a name from the list. '}
          Typed text alone isn't used.
        </p>
      )}
    </div>
  )
}
