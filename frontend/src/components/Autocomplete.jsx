import { useEffect, useId, useState } from 'react'

const MIN_CHARS = 2
const DELAY_MS = 250

// A search box over one of the API's name searches. Only a name picked from the list is passed to onSelect.
export default function Autocomplete({ label, placeholder, search, onSelect, exclude = [], disabled = false }) {
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

  const shown = options.filter((o) => !exclude.includes(o.id))

  function pick(option) {
    onSelect(option)
    setText('')
    setOptions([])
    setOpen(false)
  }

  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, shown.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && open && active >= 0 && shown[active]) {
      e.preventDefault()
      pick(shown[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const listId = `${id}-list`
  const showList = open && text.trim().length >= MIN_CHARS && status !== 'idle'

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
          {status === 'done' && shown.length === 0 && <li className="option-note">No matches</li>}
          {shown.map((o, i) => (
            <li
              key={o.id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'option active' : 'option'}
              onMouseDown={(e) => e.preventDefault()} // keep focus so the click lands before blur closes the list
              onClick={() => pick(o)}
            >
              <span className="option-name">{o.name}</span>
              <span className="option-meta">
                {o.n_films} {o.n_films === 1 ? 'film' : 'films'}{o.known_for ? ` · ${o.known_for}` : ''}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
