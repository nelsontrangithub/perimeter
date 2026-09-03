import { useEffect, useState } from "react";
import { api, type ApiClient } from "../api/client";
import type { paths } from "../api/schema";

type IndexHealth = paths["/admin/api/index"]["get"]["responses"]["200"]["content"]["application/json"];

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
}

export function IndexPage({ client = api }: { client?: ApiClient }) {
  const [health, setHealth] = useState<IndexHealth | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    client.GET("/admin/api/index").then(({ data, error }) => {
      if (cancelled) return;
      if (error || !data) setError("could not load index health");
      else setHealth(data);
    });
    return () => {
      cancelled = true;
    };
  }, [client]);

  if (error) return <p role="alert">{error}</p>;
  if (!health) return <p>Loading…</p>;
  const cache = health.acl_cache;
  const cacheTotal = cache.hits + cache.misses;
  return (
    <section>
      <h2>Index</h2>
      <p className="hint">
        The index holds vectors, chunk IDs, and ACL rows. It contains no document text (ADR-006).
      </p>
      <div className="cols">
        <dl className="facts">
          <dt>Rows</dt>
          <dd>{health.rows}</dd>
          <dt>Staged (unflushed)</dt>
          <dd>{health.staged}</dd>
          <dt>Dimension</dt>
          <dd>{health.dimension}</dd>
          <dt>Quantizer</dt>
          <dd>{health.quantizer_fitted ? "fitted" : "not fitted (empty)"}</dd>
          <dt>Rescore multiplier</dt>
          <dd>{health.rescore_multiplier}×</dd>
          <dt>ACL principals</dt>
          <dd>{health.acl_principals}</dd>
          <dt>Documents / chunks in store</dt>
          <dd>
            {health.documents} / {health.chunks}
          </dd>
          <dt>Bytes on disk</dt>
          <dd>{bytes(health.bytes_on_disk)}</dd>
          <dt>Bytes per chunk</dt>
          <dd data-testid="bytes-per-chunk">{health.bytes_per_chunk.toFixed(0)}</dd>
        </dl>
        <div>
          <h3>Files</h3>
          <table className="grid">
            <tbody>
              {Object.entries(health.files).map(([name, size]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="num">{bytes(size)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>ACL cache</h3>
          <dl className="facts">
            <dt>TTL</dt>
            <dd>{cache.ttl_seconds}s (security parameter, ADR-004)</dd>
            <dt>Hit rate</dt>
            <dd>{cacheTotal ? `${((100 * cache.hits) / cacheTotal).toFixed(0)}%` : "n/a"}</dd>
            <dt>Entries</dt>
            <dd>{cache.size}</dd>
            <dt>Upstream errors</dt>
            <dd>{cache.errors}</dd>
          </dl>
        </div>
      </div>
    </section>
  );
}
