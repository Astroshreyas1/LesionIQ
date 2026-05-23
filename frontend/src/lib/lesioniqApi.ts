import type { CaseRecord, UploadMetadataInput } from "../types/lesioniq";

interface AnalyzeCaseInput {
  image: File;
  previewUrl: string;
  metadata: UploadMetadataInput;
}

const apiBaseUrl =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_LESIONIQ_API_BASE_URL?.replace(/\/$/, "") ?? "";

export function resolveLesionIQArtifactUrl(url?: string, outputDirectory?: string): string | undefined {
  if (!url) return undefined;
  if (/^(https?:|blob:|data:)/i.test(url)) return url;

  if (url.startsWith("/")) return apiBaseUrl ? `${apiBaseUrl}${url}` : url;

  if (outputDirectory && /^(https?:)/i.test(outputDirectory)) {
    return new URL(url, outputDirectory.endsWith("/") ? outputDirectory : `${outputDirectory}/`).toString();
  }

  return apiBaseUrl ? `${apiBaseUrl}/${url.replace(/^\//, "")}` : `/${url.replace(/^\//, "")}`;
}

export async function runLesionIQAnalysis({ image, metadata }: AnalyzeCaseInput): Promise<CaseRecord> {
  const payload = new FormData();
  payload.append("image", image);
  payload.append("metadata", JSON.stringify(metadata));
  payload.append("slm_container", "lesioniq_ollama");
  payload.append("slm_model", "gemma3:4b-it-qat");

  const response = await fetch(`${apiBaseUrl || "/api"}/cases/analyze`, {
    method: "POST",
    headers: { "ngrok-skip-browser-warning": "true" },
    body: payload
  });

  if (!response.ok) {
    throw new Error(`LesionIQ analysis failed with status ${response.status}`);
  }

  return response.json() as Promise<CaseRecord>;
}
