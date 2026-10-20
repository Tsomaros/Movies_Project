import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import App from './App.jsx'
import BrowsePage from './pages/BrowsePage.jsx'
import MoviePage from './pages/MoviePage.jsx'
import NotFound from './pages/NotFound.jsx'
import PredictPage from './pages/PredictPage.jsx'
import './index.css'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <PredictPage /> },
      { path: 'movies', element: <BrowsePage /> },
      { path: 'movies/:id', element: <MoviePage /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
