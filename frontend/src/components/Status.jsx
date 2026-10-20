// Shared loading and error states.
export function Loading({ children = 'Loading…' }) {
  return (
    <p className="loading" role="status">
      <span className="spinner" aria-hidden="true" />
      {children}
    </p>
  )
}

export function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-box" role="alert">
      <p>{message}</p>
      {onRetry && <button type="button" className="secondary" onClick={onRetry}>Try again</button>}
    </div>
  )
}
