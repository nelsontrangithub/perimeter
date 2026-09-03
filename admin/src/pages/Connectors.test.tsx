import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeClient } from "../api/client";
import { Connectors } from "./Connectors";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("Connectors page", () => {
  it("lists connectors and runs an ingest through the typed client", async () => {
    const connectors = [
      { name: "docs", kind: "filesystem", root: "/corpus", needs_request_token: false, last_run: null },
    ];
    const run = {
      started_at: "t",
      duration_seconds: 0.2,
      documents: 6,
      chunks: 6,
      skipped_unchanged: 0,
      unreadable: 0,
      error: null,
    };
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async (input) => {
        const req = input as Request;
        if (req.method === "POST" && req.url.endsWith("/admin/api/connectors/docs/ingest")) {
          return jsonResponse(run);
        }
        return jsonResponse(connectors);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Connectors client={makeClient("http://p.test")} />);
    expect(await screen.findByText("docs")).toBeInTheDocument();
    expect(screen.getByText("/corpus")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Ingest" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("6 documents"));
    const posted = fetchMock.mock.calls.map((c) => c[0] as Request).find((r) => r.method === "POST");
    expect(posted?.url).toBe("http://p.test/admin/api/connectors/docs/ingest");
    vi.unstubAllGlobals();
  });

  it("shows the server's validation detail when creation fails", async () => {
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async (input) => {
        const req = input as Request;
        if (req.method === "POST") return jsonResponse({ detail: "filesystem root is not a directory" }, 422);
        return jsonResponse([]);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Connectors client={makeClient("http://p.test")} />);
    await screen.findByText("No connectors configured.");
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "docs" } });
    fireEvent.change(screen.getByLabelText("Root directory"), { target: { value: "/nope" } });
    fireEvent.click(screen.getByRole("button", { name: "Add connector" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("filesystem root is not a directory");
    vi.unstubAllGlobals();
  });
});
