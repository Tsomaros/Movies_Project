// Calls to the FastAPI service. VITE_API_URL points at a hosted API; in development the Vite proxy serves /api.
const BASE = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status, fields = []) {
    super(message)
    this.status = status
    this.fields = fields // [{field, message}] from a FastAPI 422
  }
}

function fieldErrors(detail) {
  if (!Array.isArray(detail)) return []
  return detail.map((d) => ({
    field: (d.loc ?? []).filter((p) => p !== 'body' && p !== 'query').join('.'),
    message: d.msg,
  }))
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(BASE + path, options)
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError('Could not reach the prediction service. Is the API running?', 0)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok && body === null && res.status >= 500) {
    // A proxy or gateway answering for a stopped API, or a crash without a JSON body.
    throw new ApiError(`The prediction service is not responding (${res.status}). Is the API running?`, res.status)
  }
  if (!res.ok) {
    const detail = body?.detail
    const fields = fieldErrors(detail)
    const message = typeof detail === 'string' ? detail
      : fields.length ? 'Some inputs were rejected.' : `The service answered with an error (${res.status}).`
    throw new ApiError(message, res.status, fields)
  }
  return body
}

export function get(path, params = {}, signal) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
  return request(query.size ? `${path}?${query}` : path, { signal })
}

export function post(path, json) {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(json) })
}

export const searchPeople = (role) => (q, signal) => get('/people', { q, role }, signal)
export const searchCompanies = (q, signal) => get('/companies', { q }, signal)
