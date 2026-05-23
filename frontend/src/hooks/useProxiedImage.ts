import { useEffect, useState } from "react";

/**
 * Fetches an image URL via `fetch()` with custom headers and returns a blob URL.
 *
 * This is required because ngrok free-tier injects a browser warning interstitial
 * page on direct `<img src>` requests. By fetching with the
 * `ngrok-skip-browser-warning` header, we bypass the interstitial and get the
 * actual image bytes, which are then served as a local blob URL.
 *
 * For non-ngrok URLs (localhost, blob:, data:) the original URL is returned as-is.
 */
export function useProxiedImage(url: string | undefined): string | undefined {
  const [blobUrl, setBlobUrl] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!url) {
      setBlobUrl(undefined);
      return;
    }

    // Only proxy URLs that go through ngrok (they need the header).
    // Local, blob, and data URLs can be used directly.
    const needsProxy = url.includes(".ngrok-free.") || url.includes(".ngrok.");
    if (!needsProxy) {
      setBlobUrl(url);
      return;
    }

    let revoke: string | null = null;
    let cancelled = false;

    fetch(url, {
      headers: { "ngrok-skip-browser-warning": "true" },
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Image fetch failed: ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const objectUrl = URL.createObjectURL(blob);
        revoke = objectUrl;
        setBlobUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setBlobUrl(url); // fallback to raw URL on error
      });

    return () => {
      cancelled = true;
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [url]);

  return blobUrl;
}
