import { describe, expect, it, vi } from "vitest";
import { makeClient } from "./client";

describe("typed api client", () => {
  it("calls /health and returns the typed payload", async () => {
    const payload = {
      status: "ok",
      version: "0.1.0",
      documents: 1,
      chunks: 2,
      index_size: 2,
      embedder: "local",
      store: "memory",
      reranker: "none",
    };
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => new Response(JSON.stringify(payload), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = makeClient("http://perimeter.test");
    const { data, error } = await client.GET("/health");
    expect(error).toBeUndefined();
    expect(data?.index_size).toBe(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const request = fetchMock.mock.calls[0]?.[0] as Request;
    expect(request.url).toBe("http://perimeter.test/health");
    vi.unstubAllGlobals();
  });

  it("posts the ACL invalidation hook with a typed body", async () => {
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async () => new Response(JSON.stringify({ invalidated: "alice" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = makeClient("http://perimeter.test");
    const { data } = await client.POST("/admin/api/acl/invalidate", { body: { principal: "alice" } });
    expect(data?.invalidated).toBe("alice");
    vi.unstubAllGlobals();
  });
});
