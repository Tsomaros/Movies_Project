import Autocomplete from './Autocomplete.jsx'
import Chip from './Chip.jsx'

export const MAX_ACTORS = 3

// Up to three lead actors, top billed first.
export default function ActorPicker({ search, value, onChange }) {
  const full = value.length >= MAX_ACTORS
  return (
    <div className="field">
      <Autocomplete
        label={`Lead actors (${value.length}/${MAX_ACTORS}, top billed first)`}
        placeholder={full ? 'Three actors picked' : 'Search actors'}
        search={search}
        exclude={value.map((a) => a.id)}
        disabled={full}
        allowNewcomer
        onSelect={(actor) => onChange([...value, actor])}
      />
      {value.length > 0 && (
        <div className="chips">
          {value.map((a, i) => (
            <Chip key={a.id} onRemove={() => onChange(value.filter((x) => x.id !== a.id))}>
              {`${i + 1}. ${a.name}${a.newcomer ? ' (newcomer)' : ''}`}
            </Chip>
          ))}
        </div>
      )}
    </div>
  )
}
