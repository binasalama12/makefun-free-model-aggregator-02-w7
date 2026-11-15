// StickyRoute Aggregator — brief reading and catalog building.
//
// The trial brief ($MAKEFUN_PROMPT_FILE) is prose with two JSON documents
// buried in it. Each document announces itself through a `schema` string
// under the "free-model-aggregator-02/" prefix: one carries the free model
// table, the other the six-prompt manifest. This module digs both documents
// out, normalises their rows, and derives the catalog document that
// `--list` prints and the register call posts (the very same object, so
// the two can never drift apart).
//
// Nothing here looks at a clock or a random source: the same brief always
// yields the same catalog, byte for byte, on a live run and on its replay.

import { readFileSync } from 'node:fs';

export type Bag = Record<string, unknown>;

export function isBag(value: unknown): value is Bag {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export interface FreeModel {
  /** Dialable gateway id, e.g. "openrouter/acme/thing:free". */
  dialId: string;
  vendor: string;
  label: string;
  /** House quality rating in milli-points; higher is better. */
  ratingMilli: number;
  contextLength: number | null;
  canTools: boolean | null;
  canVision: boolean | null;
  canJson: boolean | null;
  isFree: boolean;
}

export interface PromptSpec {
  id: string;
  wire: 'openai' | 'anthropic';
  streaming: boolean;
  weight: number;
}

export interface BriefData {
  models: FreeModel[];
  prompts: PromptSpec[];
  problem: string | null;
}

export interface CatalogDoc {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

const SCHEMA_TAG = 'free-model-aggregator-02/';
export const PUBLIC_BASE_URL = 'https://openrouter.ai/api/v1';
export const KEY_ENV_NAME = 'OPENROUTER_API_KEY';

// --- digging the JSON documents out of the prose ---------------------------

/** Index just past the object that opens at `open`, honouring quoted text. */
function objectSpan(text: string, open: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = open; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/** Every top-level JSON object in the brief that carries a string schema. */
function embeddedDocuments(text: string): Bag[] {
  const found: Bag[] = [];
  let cursor = text.indexOf('{');
  while (cursor !== -1) {
    const end = objectSpan(text, cursor);
    if (end !== -1) {
      try {
        const parsed: unknown = JSON.parse(text.slice(cursor, end));
        if (isBag(parsed) && typeof parsed.schema === 'string') found.push(parsed);
      } catch {
        // Prose brace, not a document — keep scanning.
      }
    }
    cursor = text.indexOf('{', cursor + 1);
  }
  return found;
}

function pickDocument(docs: Bag[], keys: string[]): { doc: Bag; key: string } | null {
  for (const doc of docs) {
    const schema = typeof doc.schema === 'string' ? doc.schema : '';
    if (!schema.includes(SCHEMA_TAG)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

// --- row normalisation ------------------------------------------------------

function asText(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asFlag(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function toModel(raw: unknown): FreeModel | null {
  if (!isBag(raw)) return null;
  const bare = asText(raw.id) || asText(raw.model);
  if (bare === '') return null;
  const vendor = asText(raw.provider) || asText(raw.wire) || 'openrouter';
  // Rows that spell the bare upstream id get the gateway prefix restored.
  const dialId = vendor === 'openrouter' && !bare.startsWith('openrouter/') ? `openrouter/${bare}` : bare;
  return {
    dialId,
    vendor,
    label: asText(raw.display_name) || bare,
    ratingMilli: asNumber(raw.rating_milli) ?? 0,
    contextLength: asNumber(raw.context_length),
    canTools: asFlag(raw.tools),
    canVision: asFlag(raw.vision),
    canJson: asFlag(raw.json),
    isFree: raw.free !== false,
  };
}

function toPrompt(raw: unknown): PromptSpec | null {
  if (!isBag(raw)) return null;
  const id = asText(raw.id);
  if (id === '') return null;
  return {
    id,
    wire: raw.api === 'anthropic' ? 'anthropic' : 'openai',
    streaming: raw.stream === true,
    weight: raw.weight === 2 ? 2 : 1,
  };
}

/** Parse the brief file; on any trouble, return what was readable plus a note. */
export function loadBrief(path: string): BriefData {
  if (path === '') return { models: [], prompts: [], problem: 'MAKEFUN_PROMPT_FILE is unset' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], problem: `cannot read brief: ${(err as Error).message}` };
  }
  const docs = embeddedDocuments(text);
  const modelHit = pickDocument(docs, ['models', 'ratings']);
  if (modelHit === null) return { models: [], prompts: [], problem: 'brief holds no model document' };
  const models = (modelHit.doc[modelHit.key] as unknown[])
    .map(toModel)
    .filter((m): m is FreeModel => m !== null);
  const promptHit = pickDocument(docs, ['prompts', 'manifest']);
  if (promptHit === null) return { models, prompts: [], problem: 'brief holds no prompt document' };
  const prompts = (promptHit.doc[promptHit.key] as unknown[])
    .map(toPrompt)
    .filter((p): p is PromptSpec => p !== null);
  return { models, prompts, problem: null };
}

/** The dialable set: free rows only, brief order, duplicates dropped. */
export function dialableModels(brief: BriefData): FreeModel[] {
  const seen = new Set<string>();
  const out: FreeModel[] = [];
  for (const model of brief.models) {
    if (!model.isFree || seen.has(model.dialId)) continue;
    seen.add(model.dialId);
    out.push(model);
  }
  return out;
}

/**
 * The catalog document. base_url is OpenRouter's public endpoint on
 * purpose — never the trial gateway's address, which differs between a
 * live run and its replay while this document must not.
 */
export function catalogFrom(brief: BriefData): CatalogDoc {
  const models = dialableModels(brief).map((m) => ({
    id: m.dialId,
    free: true,
    context_length: m.contextLength,
  }));
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{
      id: 'openrouter',
      base_url: PUBLIC_BASE_URL,
      key_env: KEY_ENV_NAME,
      wire: 'openai',
      models,
    }],
  };
}
