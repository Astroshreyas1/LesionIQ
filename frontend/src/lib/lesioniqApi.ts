import type { CaseRecord, UploadMetadataInput } from "../types/lesioniq";

interface AnalyzeCaseInput {
  image: File;
  previewUrl: string;
  metadata: UploadMetadataInput;
}

// Always use relative /api path — routes through Vercel rewrite in prod,
// and through vite.config.ts proxy in local dev. No env var needed.
const API_BASE = "/api";
const ARTIFACTS_BASE = "/artifacts";

export function resolveLesionIQArtifactUrl(url?: string, outputDirectory?: string): string | undefined {
  if (!url) return undefined;
  if (/^(https?:|blob:|data:)/i.test(url)) return url;

  if (url.startsWith("/artifacts")) return url;
  if (url.startsWith("/")) return url;

  if (outputDirectory && /^(https?:)/i.test(outputDirectory)) {
    return new URL(url, outputDirectory.endsWith("/") ? outputDirectory : `${outputDirectory}/`).toString();
  }

  return `/${url.replace(/^\//, "")}`;
}

export async function runLesionIQAnalysis({ image, metadata }: AnalyzeCaseInput): Promise<CaseRecord> {
  const payload = new FormData();
  payload.append("image", image);
  payload.append("metadata", JSON.stringify(metadata));
  payload.append("slm_container", "lesioniq_ollama");
  payload.append("slm_model", "gemma3:4b-it-qat");

  const response = await fetch(`${API_BASE}/cases/analyze`, {
    method: "POST",
    body: payload
  });

  if (!response.ok) {
    throw new Error(`LesionIQ analysis failed with status ${response.status}`);
  }

  return response.json() as Promise<CaseRecord>;
}
