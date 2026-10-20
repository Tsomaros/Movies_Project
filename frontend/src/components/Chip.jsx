export default function Chip({ children, onRemove }) {
  return (
    <span className="chip">
      {children}
      <button type="button" className="chip-remove" onClick={onRemove} aria-label={`Remove ${children}`}>×</button>
    </span>
  )
}
