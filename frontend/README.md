# Frontend

React (Vite) app with the creator form. It needs Node 22 LTS and the API running on port 8000 (see [backend/README.md](../backend/README.md)).

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173. In development, Vite forwards `/api/*` to the API at `http://127.0.0.1:8000`. To use an API hosted elsewhere, set `VITE_API_URL` before building:

```bash
npm run lint
npm run build
```

| route | page |
|---|---|
| `/` | creator form: genres, runtime, release month and year, language, director, lead actors, company, optional budget and overview. The results show below it: budget, revenue and score ranges with a range chart, the top 5 factors in plain English, a "try changing one thing" comparison (genres, release month or year, runtime, language or budget) and a note on the model's limitations. People not in our data can be added as newcomers |
| `/movies` | public browse page: a poster grid with title search, genre and release-year filters, and pages of 24. The filters are kept in the URL |
| `/movies/:id` | movie detail: poster, overview, facts (release date, language, budget, box office, score), key crew, companies and the top 10 cast |

The footer shows TMDB's attribution notice and logo. `public/tmdb-logo.svg` is TMDB's official short logo from https://www.themoviedb.org/about/logos-attribution.
