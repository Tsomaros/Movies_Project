import { NavLink, Outlet } from 'react-router-dom'

export default function App() {
  return (
    <div className="layout">
      <header className="site-header">
        <NavLink to="/" className="brand">Movie Prediction</NavLink>
        <nav>
          <NavLink to="/" end>Predict</NavLink>
          <NavLink to="/movies">Browse</NavLink>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="site-footer">
        This product uses the TMDB API but is not endorsed or certified by TMDB.
      </footer>
    </div>
  )
}
