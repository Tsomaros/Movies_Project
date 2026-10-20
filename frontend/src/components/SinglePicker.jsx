import Autocomplete from './Autocomplete.jsx'
import Chip from './Chip.jsx'

// One person or company: the search box until a match is picked, then a removable chip.
export default function SinglePicker({ label, placeholder, search, value, onChange }) {
  if (value) {
    return (
      <div className="field">
        <span className="label">{label}</span>
        <div className="chips"><Chip onRemove={() => onChange(null)}>{value.name}</Chip></div>
      </div>
    )
  }
  return (
    <div className="field">
      <Autocomplete label={label} placeholder={placeholder} search={search} onSelect={onChange} />
    </div>
  )
}
