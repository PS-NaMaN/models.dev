import { readdirSync } from "node:fs";
import path from "node:path";
import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedModel } from "../index.js";
import { factorBaseModel } from "./openrouter.js";

const API_ENDPOINT = "https://api.xkiro.com/v1/models";
const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");

// xKiro vendor prefix -> models.dev lab directory.
const VENDOR_TO_LAB: Record<string, string> = {
  qwen: "alibaba",
  "z-ai": "zhipuai",
  minimax: "minimax",
  deepseek: "deepseek",
  anthropic: "anthropic",
  openai: "openai",
  mistralai: "mistral",
  google: "google",
  nvidia: "nvidia",
  moonshotai: "moonshotai",
  "x-ai": "xai",
  sensenova: "sensenova",
  tencent: "tencent",
  xiaomi: "xiaomi",
  meta: "meta",
};

// Served IDs that do not derive from any lab filename rule. Verified against
// vendor docs; prefer adding a rule over growing this map.
const BASE_MODEL_ALIASES: Record<string, string> = {
  "deepseek/deepseek-chat-v3.1": "deepseek/deepseek-v3.1",
  "mistralai/ministral-8b": "mistral/ministral-3-8b-instruct-2512",
  "mistralai/mistral-medium-3.5": "mistral/mistral-medium-2604",
  "meta/muse-spark-1.2-contributor": "meta/muse-spark-1.2",
  "nvidia/nemotron-3-nano": "nvidia/nemotron-3-nano-30b-a3b",
  "nvidia/nemotron-3-nano-omni": "nvidia/nemotron-3-nano-omni-30b-a3b",
  "nvidia/nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
  "nvidia/nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
  "nvidia/llama-3.3-nemotron-super-49b": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "z-ai/glm-4.5-airx": "zhipuai/glm-4.5-air",
  "z-ai/glm-4.5-x": "zhipuai/glm-4.5",
  "z-ai/glm-4.6v-flashx": "zhipuai/glm-4.6v-flash",
};

// Words that describe a two-position on/off control, not an effort scale.
// The catalog schema only allows none/minimal/low/medium/high/xhigh/max/default
// as effort values, so these must map to a toggle instead.
const TOGGLE_WORDS = new Set(["off", "on", "adaptive", "disabled"]);

type EffortValue =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max"
  | "default";

const SCHEMA_EFFORT = new Set<string>([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "default",
]);

type Modality = "text" | "audio" | "image" | "video" | "pdf";

const Pricing = z
  .object({
    input: z.number(),
    output: z.number(),
    cache_read: z.number().optional(),
    cache_write: z.number().optional(),
  })
  .passthrough();

const Capabilities = z
  .object({
    vision: z.boolean(),
    tools: z.boolean(),
    reasoning: z.boolean(),
  })
  .passthrough();

const ReasoningEfforts = z
  .object({
    levels: z.array(z.string()),
    default: z.string().optional(),
  })
  .passthrough();

export const XkiroModel = z
  .object({
    id: z.string(),
    display_name: z.string().optional(),
    access_tier: z.string().optional(),
    pricing: Pricing.optional(),
    capabilities: Capabilities.optional(),
    context_length: z.number().optional(),
    max_output_tokens: z.number().optional(),
    reasoning_efforts: ReasoningEfforts.optional(),
  })
  .passthrough();

export const XkiroResponse = z
  .object({
    data: z.array(XkiroModel),
  })
  .passthrough();

export type XkiroModel = z.infer<typeof XkiroModel>;

export const xkiro = {
  id: "xkiro",
  name: "xKiro",
  modelsDir: "providers/xkiro/models",
  preserveBaseModels: false,
  async fetchModels() {
    // Public catalog, no auth. A browser-style UA: bare script clients are
    // filtered at the edge.
    const response = await fetch(API_ENDPOINT, {
      headers: { "User-Agent": "models-dev-sync (+https://models.dev)" },
    });
    if (!response.ok) {
      throw new Error(
        `xKiro models request failed: ${response.status} ${response.statusText}`,
      );
    }
    return response.json();
  },
  parseModels(raw) {
    return XkiroResponse.parse(raw).data;
  },
  translateModel(model, context) {
    // Stealth previews come and go faster than docs; never author them.
    if (model.id.startsWith("stealth/")) return undefined;
    const base = resolveBaseModel(model.id);
    const translated = buildXkiroModel(model, base, context.existing(model.id));
    return {
      id: model.id,
      model: translated,
      header: fileHeader(translated),
    };
  },
} satisfies SyncProvider<XkiroModel>;

export function buildXkiroModel(
  model: XkiroModel,
  base: string,
  existing: ExistingModel | undefined,
): SyncedModel {
  const caps = model.capabilities;
  if (caps === undefined) {
    throw new Error(`xkiro: ${model.id} has no capabilities`);
  }
  const reasoning = caps.reasoning;
  const input: Modality[] = caps.vision ? ["text", "image"] : ["text"];

  const context = model.context_length ?? existing?.limit?.context;
  if (context === undefined) {
    throw new Error(`xkiro: ${model.id} has no context length`);
  }
  const output = model.max_output_tokens ?? existing?.limit?.output;
  if (output === undefined) {
    throw new Error(`xkiro: ${model.id} has no output limit`);
  }
  const limit = { context, input: existing?.limit?.input, output };

  const pricing = model.pricing;
  // OVHcloud precedent: free models carry no [cost] section.
  const cost =
    pricing === undefined
      ? existing?.cost
      : pricing.input === 0 && pricing.output === 0
        ? undefined
        : {
            input: pricing.input,
            output: pricing.output,
            cache_read: pricing.cache_read,
            cache_write: pricing.cache_write,
          };

  const reasoningOptions = !reasoning
    ? undefined
    : mapReasoningOptions(model.id, model.reasoning_efforts?.levels);

  const name = model.display_name ?? humanizeModelName(model.id);

  return factorBaseModel(
    base,
    {
      name,
      attachment: caps.vision,
      reasoning,
      tool_call: caps.tools,
      modalities: { input, output: ["text"] as Modality[] },
      limit,
      cost,
      reasoning_options: reasoningOptions,
    },
    limit,
    existing?.base_model_omit,
  );
}

function mapReasoningOptions(id: string, levels: string[] | undefined) {
  if (levels === undefined || levels.length === 0) {
    // Reasoning capability with no caller control on this host (e.g. Qwen:
    // reasoning parameters are ignored). Empty means no control, not uncertainty.
    return [];
  }
  const graded = levels.filter((level) => !TOGGLE_WORDS.has(level));
  for (const level of graded) {
    if (!SCHEMA_EFFORT.has(level)) {
      throw new Error(`xkiro: ${id} has unknown reasoning level ${level}`);
    }
  }
  if (graded.length === 0) {
    // Two-position on/off control (off/on, adaptive/disabled).
    return [{ type: "toggle" as const }];
  }
  return [
    { type: "effort" as const, values: graded as EffortValue[] },
  ];
}

function resolveBaseModel(id: string): string {
  const alias = BASE_MODEL_ALIASES[id];
  if (alias !== undefined) {
    if (!canonicalExists(alias)) {
      throw new Error(`xkiro: alias target missing ${alias} for ${id}`);
    }
    return alias;
  }
  const slash = id.indexOf("/");
  if (slash === -1) throw new Error(`xkiro: bad id ${id}`);
  const lab = VENDOR_TO_LAB[id.slice(0, slash)];
  if (lab === undefined) throw new Error(`xkiro: unknown vendor for ${id}`);
  let model = id.slice(slash + 1);
  if (model.endsWith(":free")) model = model.slice(0, -":free".length);
  const candidates = [
    model,
    model.toLowerCase(),
    model.replace(/\./g, "-"),
    model.replace(/-\d{4}-\d{2}-\d{2}$/, ""),
    `${model}-latest`,
    `${model}-preview`,
  ];
  for (const candidate of new Set(candidates)) {
    const hit = canonicalIn(lab, candidate);
    if (hit !== undefined) return `${lab}/${hit}`;
  }
  throw new Error(`xkiro: no base_model for ${id}`);
}

function canonicalExists(candidate: string): boolean {
  const slash = candidate.indexOf("/");
  if (slash === -1) return false;
  return (
    canonicalIn(candidate.slice(0, slash), candidate.slice(slash + 1)) !==
    undefined
  );
}

// existsSync is case-insensitive on Windows/macOS; verify the real on-disk
// filename case so the resolved base_model matches the canonical metadata
// exactly (and CI on Linux). Exact case wins over case-insensitive matches.
function canonicalIn(lab: string, name: string): string | undefined {
  let files: string[];
  try {
    files = readdirSync(path.join(MODELS_DIR, lab));
  } catch {
    return undefined;
  }
  const exact = `${name}.toml`;
  if (files.includes(exact)) return name;
  const want = exact.toLowerCase();
  const hit = files.find((file) => file.toLowerCase() === want);
  return hit === undefined ? undefined : hit.slice(0, -".toml".length);
}

function fileHeader(model: SyncedModel): string | undefined {
  const sources =
    "# Sources: https://api.xkiro.com/v1/models (pricing, limits, capabilities)\n";
  if (
    model.reasoning_options?.some((option) => option.type === "toggle") ===
    true
  ) {
    return (
      "# Toggle: reasoning_effort none/disabled intent, translated by xKiro\n" +
      "# to the model's own off switch\n" +
      "# https://docs.xkiro.com/guides/reasoning/\n" +
      sources
    );
  }
  return sources;
}

function humanizeModelName(modelId: string): string {
  const modelPart = modelId.split("/").at(-1) ?? modelId;
  return modelPart.replace(/[-_:]/g, " ");
}
