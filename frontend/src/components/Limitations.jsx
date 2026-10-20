// What the predictions can and can't tell you. The numbers come from the README's evaluation.
export default function Limitations() {
  return (
    <aside className="limitations" aria-labelledby="limitations-title">
      <h3 id="limitations-title">Limitations: read before you rely on these numbers</h3>
      <ul>
        <li>
          Each range is the model's 10–90% range. About 8 in 10 similar past films fell inside it, so roughly
          1 in 5 films lands outside, above or below.
        </li>
        <li>
          The model learned from TMDB films released 1980–2025 that report both a budget and revenue, mostly larger
          and US releases. Revenue is worldwide box office in 2024 dollars; streaming and home video aren't counted.
        </li>
        <li>
          It can't see marketing, reviews, competing releases or the quality of the script. Anyone not in our data
          counts as a newcomer.
        </li>
        <li>The factors show what the model leans on, not cause and effect.</li>
        <li>
          Recent films have scored higher than the model expects, and the 2020–21 cinema closures make box office
          for those years less reliable.
        </li>
      </ul>
    </aside>
  )
}
