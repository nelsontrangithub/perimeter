import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeClient } from "../api/client";
import { Simulator } from "./Simulator";

describe("Simulator page", () => {
  it("posts the identity and renders visibility with reasons", async () => {
    const simulation = {
      principal: "carol@example.com",
      effective_principals: ["carol@example.com", "contractors", "everyone", "sre"],
      documents: [
        { id: "a", title: "onboarding.md", uri: "file:///a", connector: "filesystem", grants: ["everyone"], denies: [], visible: true, reason: "granted via everyone" },
        { id: "b", title: "index-design.md", uri: "file:///b", connector: "filesystem", grants: ["eng"], denies: ["contractors"], visible: false, reason: "denied via contractors" },
      ],
      visible_count: 1,
      total: 2,
      results: null,
    };
    let body: unknown = null;
    const fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(
      async (input) => {
        body = JSON.parse((await (input as Request).text()) || "null");
        return new Response(JSON.stringify(simulation), { status: 200, headers: { "content-type": "application/json" } });
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Simulator client={makeClient("http://p.test")} />);
    fireEvent.click(screen.getByRole("button", { name: "Simulate" }));
    expect(await screen.findByRole("status")).toHaveTextContent("1 of 2");
    expect(screen.getByText("denied via contractors")).toBeInTheDocument();
    expect(screen.getByText("granted via everyone")).toBeInTheDocument();
    expect(body).toEqual({ principal: "carol@example.com", groups: ["sre", "contractors"], query: null, k: 5 });
    vi.unstubAllGlobals();
  });
});
