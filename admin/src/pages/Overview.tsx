import { useEffect, useState } from "react";
import { api, type Health } from "../api/client";

export function Overview() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .GET("/health")
      .then(({ data, error }) => {
        if (cancelled) return;
        if (error || !data) setError("health check failed");
        else setHealth(data);
      })
      .catch(() => !cancelled && setError("health check failed"));
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p role="alert">{error}</p>;
  if (!health) return <p>Loading…</p>;
  return (
    <section>
      <h2>Overview</h2>
      <dl className="facts">
        <dt>Status</dt>
        <dd>{health.status}</dd>
        <dt>Version</dt>
        <dd>{health.version}</dd>
        <dt>Documents</dt>
        <dd>{health.documents}</dd>
        <dt>Chunks</dt>
        <dd>{health.chunks}</dd>
        <dt>Index rows</dt>
        <dd>{health.index_size}</dd>
        <dt>Embedder</dt>
        <dd>{health.embedder}</dd>
        <dt>Store</dt>
        <dd>{health.store}</dd>
        <dt>Reranker</dt>
        <dd>{health.reranker}</dd>
      </dl>
    </section>
  );
}
