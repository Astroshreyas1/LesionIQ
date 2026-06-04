// Vercel Edge Function — proxies /artifacts/* requests to the ngrok-hosted
// backend with the `ngrok-skip-browser-warning` header set, so the free-tier
// ngrok interstitial HTML doesn't get returned in place of the image bytes.
//
// Routing:
//   Browser  GET /artifacts/CASE_ID/raw.png
//      ↓ vercel.json rewrite
//   Vercel   /api/artifacts/CASE_ID/raw.png   (this function)
//      ↓ this fetch
//   ngrok    https://<tunnel>/artifacts/CASE_ID/raw.png
//      ↑ no interstitial because the bypass header is set
//
// Edge runtime: low latency, no cold-start surcharge.

export const config = { runtime: "edge" };

const NGROK_BASE = "https://overload-opacity-connector.ngrok-free.dev";

export default async function handler(request: Request): Promise<Response> {
  const incoming = new URL(request.url);
  // Strip the leading /api → forward /artifacts/CASE_ID/raw.png upstream
  const targetPath = incoming.pathname.replace(/^\/api/, "");
  const targetUrl = `${NGROK_BASE}${targetPath}${incoming.search}`;

  let upstream: Response;
  try {
    upstream = await fetch(targetUrl, {
      method: "GET",
      headers: {
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "LesionIQ-Vercel-Proxy/1.0"
      }
    });
  } catch (err) {
    return new Response(
      `Artifact proxy upstream error: ${err instanceof Error ? err.message : String(err)}`,
      { status: 502, headers: { "Content-Type": "text/plain" } }
    );
  }

  // Pass through the headers that matter for image rendering / caching.
  const responseHeaders = new Headers();
  ["content-type", "content-length", "last-modified", "etag"].forEach((h) => {
    const v = upstream.headers.get(h);
    if (v) responseHeaders.set(h, v);
  });
  responseHeaders.set("Cache-Control", "public, max-age=3600, immutable");

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders
  });
}
