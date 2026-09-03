import createClient from "openapi-fetch";
import type { paths } from "./schema";

/**
 * Typed API client generated from the server's OpenAPI document.
 * `npm run generate` regenerates `schema.d.ts`; CI fails if it is stale, so a
 * route or field the server does not have cannot be referenced from here.
 */
export type ApiClient = ReturnType<typeof createClient<paths>>;

export function makeClient(baseUrl = ""): ApiClient {
  return createClient<paths>({ baseUrl });
}

export const api: ApiClient = makeClient();

export type Health = paths["/health"]["get"]["responses"]["200"]["content"]["application/json"];
