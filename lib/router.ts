// StickyRoute Aggregator — the routing brain.
//
// Two ideas live here:
//
//  * Stickiness. Prompts come in kinds, keyed by (wire, streaming, needs
//    JSON, needs tools, needs vision). When a model answers a kind well,
//    we remember it and lead with it the next time that kind shows up —
//    but only while its rating still sits in the top band for that kind,
//    so a sticky favourite can never drag quality down. A success is the
//    only thing that writes the cache; a refusal never promotes anyone.
//
//  * Cooldowns. A model the gateway refuses (429, 5xx, network) sits out
//    the next few routing rounds. The counter is a logical round number
//    that ticks once per client request — never a wall clock — so a
//    replay that sees the same answers makes the same choices.
//
// Ordering inside one request is fully deterministic: sticky pick first
// (when it qualifies), then rating descending, then dial id ascending
// as the tie-break.

import { isBag, type Bag, type FreeModel } from './brief.ts';

export interface Needs {
  tools: boolean;
  json: boolean;
  vision: boolean;
}

/** What a client request body actually asks for. */
export function needsFromOpenAIBody(body: Bag): Needs {
  const format = isBag(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    json: format === 'json_object' || format === 'json_schema',
    vision: bodyMentionsImage(body.messages),
  };
}

/** What an Anthropic-shaped body asks for, judged on its own fields. */
export function needsFromAnthropicBody(body: Bag): Needs {
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    json: false,
    vision: bodyMentionsImage(body.messages),
  };
}

function bodyMentionsImage(messages: unknown): boolean {
  if (!Array.isArray(messages)) return false;
  for (const message of messages) {
    if (!isBag(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isBag(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') return true;
      if (isBag(block.image_url)) return true;
    }
  }
  return false;
}

/** A model may serve a need only when the brief says it can, outright. */
export function eligible(model: FreeModel, needs: Needs): boolean {
  if (needs.tools && model.canTools !== true) return false;
  if (needs.json && model.canJson !== true) return false;
  if (needs.vision && model.canVision !== true) return false;
  return true;
}

/** The sticky-cache key for one prompt kind. */
export function kindKey(wire: string, streaming: boolean, needs: Needs): string {
  return [wire, streaming ? 'sse' : 'json', needs.json ? 'J' : '-', needs.tools ? 'T' : '-', needs.vision ? 'V' : '-'].join('|');
}

/** How far below the best rating a sticky pick may sit and still lead. */
const STICKY_BAND_MILLI = 150;

interface CooldownNote {
  /** Routing rounds the model still has to sit out. */
  roundsLeft: number;
  refusals: number;
}

export class StickyRouter {
  private readonly pool: FreeModel[];
  private readonly sticky = new Map<string, string>();
  private readonly cooling = new Map<string, CooldownNote>();
  private round = 0;

  constructor(pool: FreeModel[]) {
    this.pool = pool;
  }

  /** Advance the logical clock; called once per client request. */
  tick(): void {
    this.round += 1;
    for (const [id, note] of this.cooling) {
      note.roundsLeft -= 1;
      if (note.roundsLeft <= 0) this.cooling.delete(id);
    }
  }

  isCooling(id: string): boolean {
    return this.cooling.has(id);
  }

  /**
   * Candidate order for one request: the sticky model for this kind
   * first when it is eligible, not cooling, and rated within the top
   * band; then every other eligible model by rating. Cooling models are
   * held back to a last-resort tail so a request is never left with
   * nobody to dial.
   */
  orderFor(kind: string, needs: Needs): FreeModel[] {
    const fit = this.pool.filter((m) => eligible(m, needs));
    fit.sort((a, b) => (b.ratingMilli - a.ratingMilli) || (a.dialId < b.dialId ? -1 : 1));
    if (fit.length === 0) return [];
    const best = fit[0].ratingMilli;
    const head: FreeModel[] = [];
    const tail: FreeModel[] = [];
    const stickyId = this.sticky.get(kind);
    for (const model of fit) {
      if (this.isCooling(model.dialId)) tail.push(model);
      else head.push(model);
    }
    if (stickyId !== undefined) {
      const at = head.findIndex((m) => m.dialId === stickyId);
      if (at > 0 && head[at].ratingMilli >= best - STICKY_BAND_MILLI) {
        const [pick] = head.splice(at, 1);
        head.unshift(pick);
      }
    }
    return [...head, ...tail];
  }

  /** A refusal parks the model for a couple of rounds. */
  noteRefusal(id: string): void {
    const prev = this.cooling.get(id);
    this.cooling.set(id, { roundsLeft: 2, refusals: (prev?.refusals ?? 0) + 1 });
  }

  /** Only a served answer earns the sticky slot for its kind. */
  noteSuccess(kind: string, id: string): void {
    this.sticky.set(kind, id);
    this.cooling.delete(id);
  }
}
