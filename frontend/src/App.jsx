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
        <a href="https://www.themoviedb.org/" target="_blank" rel="noreferrer" className="tmdb-logo">
          <img src="/tmdb-logo.svg" alt="TMDB" width="110" height="14" />
        </a>
        <p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
      </footer>
    </div>
  )
}
