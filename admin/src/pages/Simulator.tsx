import { useState, type FormEvent } from "react";
import { api, type ApiClient } from "../api/client";
import type { paths } from "../api/schema";

type Simulation = paths["/admin/api/simulate"]["post"]["responses"]["200"]["content"]["application/json"];
type RetrievalPayload = {
  requested_k: number;
  returned: number;
  candidates: number;
  results: { text: string; score: number; citation: { title: string; uri: string; document_id: string } }[];
};

export function Simulator({ client = api }: { client?: ApiClient }) {
  const [principal, setPrincipal] = useState("carol@example.com");
  const [groups, setGroups] = useState("sre, contractors");
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { data, error } = await client.POST("/admin/api/simulate", {
      body: {
        principal: principal.trim(),
        groups: groups
          .split(",")
          .map((g) => g.trim())
          .filter(Boolean),
        query: query.trim() || null,
        k,
      },
    });
    setBusy(false);
    if (error || !data) {
      setError(typeof error?.detail === "string" ? error.detail : "simulation failed");
      return;
    }
    setSim(data);
  }

  const results = sim?.results as RetrievalPayload | null | undefined;

  return (
    <section>
      <h2>Permission simulator</h2>
      <p className="hint">
        See the corpus exactly as a principal would. Resolution uses the same resolver and cache as live
        retrieval; hidden documents are hidden for the reason the policy engine gives, and a query runs
        the real filtered scan.
      </p>
      <form onSubmit={(e) => void run(e)} className="row">
        <label>
          Principal
          <input value={principal} onChange={(e) => setPrincipal(e.target.value)} required />
        </label>
        <label>
          Forwarded groups (comma-separated)
          <input value={groups} onChange={(e) => setGroups(e.target.value)} />
        </label>
        <label>
          Query (optional)
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="revenue forecast" />
        </label>
        <label>
          k
          <input type="number" min={1} max={100} value={k} onChange={(e) => setK(Number(e.target.value))} />
        </label>
        <button type="submit" disabled={busy}>
          {busy ? "Simulating…" : "Simulate"}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {sim && (
        <>
          <p className="result" role="status">
            <strong>{sim.principal}</strong> resolves to {sim.effective_principals.join(", ")} and can see{" "}
            <strong>
              {sim.visible_count} of {sim.total}
            </strong>{" "}
            documents.
          </p>
          <table className="grid" aria-label="Documents as seen by the principal">
            <thead>
              <tr>
                <th>Document</th>
                <th>Connector</th>
                <th>Grants</th>
                <th>Denies</th>
                <th>Visible</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {sim.documents.map((d) => (
                <tr key={d.id} className={d.visible ? "visible" : "hidden-doc"}>
                  <td title={d.uri}>{d.title}</td>
                  <td>{d.connector}</td>
                  <td>{d.grants.join(", ") || "—"}</td>
                  <td>{d.denies.join(", ") || "—"}</td>
                  <td>
                    <span className={d.visible ? "pill ok" : "pill deny"}>{d.visible ? "visible" : "hidden"}</span>
                  </td>
                  <td>{d.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {results && (
            <div>
              <h3>
                Retrieval for “{query}”: {results.returned} of k={results.requested_k} returned (
                {results.candidates} permitted candidates)
              </h3>
              <ol className="results">
                {results.results.map((r) => (
                  <li key={r.citation.document_id + r.score}>
                    <div className="cite">
                      {r.citation.title} <span className="muted">score {r.score.toFixed(3)}</span>
                    </div>
                    <blockquote>{r.text}</blockquote>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </>
      )}
    </section>
  );
}
