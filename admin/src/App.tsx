import { useState } from "react";
import { Connectors } from "./pages/Connectors";
import { IndexPage } from "./pages/Index";
import { Overview } from "./pages/Overview";

type Page = "overview" | "connectors" | "index" | "simulator";

const PAGES: { id: Page; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "connectors", label: "Connectors" },
  { id: "index", label: "Index" },
  { id: "simulator", label: "Permission simulator" },
];

export function App() {
  const [page, setPage] = useState<Page>("overview");
  return (
    <div className="shell">
      <header>
        <h1>Perimeter</h1>
        <nav aria-label="Sections">
          {PAGES.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-current={page === p.id ? "page" : undefined}
              onClick={() => setPage(p.id)}
            >
              {p.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {page === "overview" && <Overview />}
        {page === "connectors" && <Connectors />}
        {page === "index" && <IndexPage />}
        {page === "simulator" && (
          <section>
            <h2>{PAGES.find((p) => p.id === page)?.label}</h2>
            <p>Coming in the next commits.</p>
          </section>
        )}
      </main>
    </div>
  );
}
