import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <section>
      <h1>Page not found</h1>
      <p><Link to="/">Back to the prediction form</Link></p>
    </section>
  )
}
