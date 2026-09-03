import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeClient } from "../api/client";
import { IndexPage } from "./Index";

describe("Index page", () => {
  it("renders layout facts, files, and cache stats", async () => {
    const health = {
      rows: 6,
      staged: 0,
      dimension: 1024,
      quantizer_fitted: true,
      rescore_multiplier: 64,
      acl_principals: 5,
      files: { "binary.bin": 768, "int8.bin": 6144 },
      bytes_on_disk: 6912,
      bytes_per_chunk: 1152,
      documents: 6,
      chunks: 6,
      acl_cache: { ttl_seconds: 60, hits: 3, misses: 1, errors: 0, evictions: 0, size: 1 },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(health), { status: 200 })),
    );
    render(<IndexPage client={makeClient("http://p.test")} />);
    expect(await screen.findByTestId("bytes-per-chunk")).toHaveTextContent("1152");
    expect(screen.getByText("binary.bin")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("64×")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
