import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, type ApiClient } from "../api/client";
import type { paths } from "../api/schema";

type ConnectorView = paths["/admin/api/connectors"]["get"]["responses"]["200"]["content"]["application/json"][number];
type IngestRun = paths["/admin/api/connectors/{name}/ingest"]["post"]["responses"]["200"]["content"]["application/json"];

export function Connectors({ client = api }: { client?: ApiClient }) {
  const [items, setItems] = useState<ConnectorView[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<"filesystem" | "gdrive">("filesystem");
  const [root, setRoot] = useState("");
  const [lastRun, setLastRun] = useState<{ name: string; run: IngestRun } | null>(null);

  const refresh = useCallback(async () => {
    const { data, error } = await client.GET("/admin/api/connectors");
    if (error || !data) setError("could not load connectors");
    else setItems(data);
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function create(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const { error } = await client.POST("/admin/api/connectors", {
      body: { name, kind, root: kind === "filesystem" ? root : null },
    });
    if (error) {
      setError(typeof error.detail === "string" ? error.detail : "could not create connector");
      return;
    }
    setName("");
    setRoot("");
    await refresh();
  }

  async function ingest(connector: string) {
    setBusy(connector);
    setError(null);
    const { data, error } = await client.POST("/admin/api/connectors/{name}/ingest", {
      params: { path: { name: connector } },
    });
    setBusy(null);
    if (error || !data) {
      setError("ingest request failed");
      return;
    }
    setLastRun({ name: connector, run: data });
    await refresh();
  }

  async function remove(connector: string) {
    await client.DELETE("/admin/api/connectors/{name}", { params: { path: { name: connector } } });
    await refresh();
  }

  return (
    <section>
      <h2>Connectors</h2>
      <p className="hint">
        Configuration holds names and paths only. A Google Drive connector receives its OAuth token per
        request in the <code>X-Perimeter-Token-gdrive</code> header and Perimeter never stores it.
      </p>
      {error && <p role="alert">{error}</p>}
      <table className="grid">
        <thead>
          <tr>
            <th>Name</th>
            <th>Kind</th>
            <th>Source</th>
            <th>Last run</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {items.length === 0 && (
            <tr>
              <td colSpan={5} className="muted">
                No connectors configured.
              </td>
            </tr>
          )}
          {items.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.kind}</td>
              <td>{c.kind === "filesystem" ? c.root : "token per request"}</td>
              <td>
                {c.last_run
                  ? c.last_run.error
                    ? `error: ${c.last_run.error}`
                    : `${c.last_run.documents} docs, ${c.last_run.chunks} chunks, ${c.last_run.skipped_unchanged} unchanged`
                  : "never"}
              </td>
              <td className="actions">
                <button type="button" disabled={busy === c.name} onClick={() => void ingest(c.name)}>
                  {busy === c.name ? "Ingesting…" : "Ingest"}
                </button>
                <button type="button" className="danger" onClick={() => void remove(c.name)}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {lastRun && (
        <p className="result" role="status">
          {lastRun.name}: {lastRun.run.documents} documents, {lastRun.run.chunks} chunks,{" "}
          {lastRun.run.skipped_unchanged} unchanged, {lastRun.run.unreadable} unreadable in{" "}
          {lastRun.run.duration_seconds}s{lastRun.run.error ? ` (${lastRun.run.error})` : ""}
        </p>
      )}
      <form onSubmit={(e) => void create(e)} className="row">
        <label>
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} required pattern="[a-z0-9][a-z0-9_-]*" />
        </label>
        <label>
          Kind
          <select value={kind} onChange={(e) => setKind(e.target.value as "filesystem" | "gdrive")}>
            <option value="filesystem">filesystem</option>
            <option value="gdrive">gdrive</option>
          </select>
        </label>
        {kind === "filesystem" && (
          <label>
            Root directory
            <input value={root} onChange={(e) => setRoot(e.target.value)} required />
          </label>
        )}
        <button type="submit">Add connector</button>
      </form>
    </section>
  );
}
