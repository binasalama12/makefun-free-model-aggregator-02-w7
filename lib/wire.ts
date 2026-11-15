// StickyRoute Aggregator — wire translation between the two client APIs
// and the single upstream API.
//
// Clients speak OpenAI Chat Completions or Anthropic Messages. The
// gateway only speaks Chat Completions, so:
//
//   OpenAI client  -> body passes through (model swapped), answer too.
//   Anthropic client -> request rewritten to Chat Completions, answer
//                       rewritten back to a Messages object or to the
//                       Anthropic SSE event sequence.
//
// One rule rules everything: the `model` field on every answer and every
// streamed chunk is the id we actually dialled. Providers spell a free
// model without the "openrouter/" prefix, so the dialled id is stamped
// back on here — never invented, always the id from the brief that went
// into the gateway request.
//
// All functions are pure in their inputs (fixed timestamps in the model
// lists included), so live and replay runs emit identical bytes.

import { isBag, type Bag } from './brief.ts';

// --- model listings ----------------------------------------------------------

// Fixed stamps: a listing that read the clock would differ between runs.
const LIST_CREATED = 1767225600;
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';

export interface ListedModel {
  id: string;
  vendor: string;
  label: string;
}

export function openAIList(models: readonly ListedModel[]): Bag {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: LIST_CREATED, owned_by: m.vendor })),
  };
}

export function anthropicList(models: readonly ListedModel[]): Bag {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.label, created_at: LIST_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

// --- errors ------------------------------------------------------------------

export function openAIErrorBody(message: string, type: string, code: string | null): Bag {
  return { error: { message, type, param: null, code } };
}

export function anthropicErrorBody(type: string, message: string): Bag {
  return { type: 'error', error: { type, message } };
}

// --- small shared helpers ----------------------------------------------------

function num(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function plainText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isBag(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function parseArgs(value: unknown): Bag {
  if (isBag(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isBag(parsed)) return parsed;
    } catch {
      // fall through to empty
    }
  }
  return {};
}

/** OpenAI finish_reason -> Anthropic stop_reason. */
export function mapStopReason(finish: unknown): string {
  switch (finish) {
    case 'length': return 'max_tokens';
    case 'tool_calls':
    case 'function_call': return 'tool_use';
    case 'content_filter': return 'refusal';
    default: return 'end_turn';
  }
}

// --- Anthropic request -> Chat Completions request ---------------------------

function imagePart(block: Bag): Bag | null {
  if (!isBag(block.source)) return null;
  const src = block.source;
  if (src.type === 'base64' && typeof src.data === 'string') {
    const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } };
  }
  if (src.type === 'url' && typeof src.url === 'string') {
    return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = plainText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

export function toChatBody(body: Bag, modelId: string): Bag {
  const messages: Bag[] = [];
  const systemText = plainText(body.system);
  if (systemText !== '') messages.push({ role: 'system', content: systemText });

  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of incoming) {
    if (!isBag(raw)) continue;
    const isAssistant = raw.role === 'assistant';
    if (typeof raw.content === 'string') {
      messages.push({ role: isAssistant ? 'assistant' : 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (isAssistant) {
      const calls: Bag[] = [];
      for (const block of raw.content) {
        if (!isBag(block) || block.type !== 'tool_use') continue;
        calls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isBag(block.input) ? block.input : {}),
          },
        });
      }
      const text = plainText(raw.content);
      messages.push({
        role: 'assistant',
        content: text === '' ? null : text,
        ...(calls.length > 0 ? { tool_calls: calls } : {}),
      });
      continue;
    }
    // A user turn: tool results become tool messages, the rest content parts.
    const parts: Bag[] = [];
    for (const block of raw.content) {
      if (!isBag(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: resultText(block.content),
        });
      } else if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image') {
        const part = imagePart(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length === 1 && parts[0].type === 'text') {
      messages.push({ role: 'user', content: parts[0].text as string });
    } else if (parts.length > 0) {
      messages.push({ role: 'user', content: parts });
    }
  }

  const out: Bag = { model: modelId, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.top_k === 'number') out.top_k = body.top_k;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Bag[] = [];
    for (const raw of body.tools) {
      if (!isBag(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isBag(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  if (isBag(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// --- Chat Completions answer -> Anthropic answer -----------------------------

export function toMessageBody(answer: Bag, modelId: string): Bag {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const first = isBag(choices[0]) ? choices[0] : {};
  const message = isBag(first.message) ? first.message : {};
  const blocks: Bag[] = [];
  const text = typeof message.content === 'string' ? message.content : plainText(message.content);
  if (text !== '') blocks.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isBag(call) || !isBag(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_1',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parseArgs(call.function.arguments),
    });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  const usage = isBag(answer.usage) ? answer.usage : {};
  return {
    id: typeof answer.id === 'string' ? `msg_${answer.id.replace(/^chatcmpl-/, '')}` : 'msg_stickyroute',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: blocks,
    stop_reason: mapStopReason(first.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: num(usage.prompt_tokens),
      output_tokens: num(usage.completion_tokens),
    },
  };
}

/** Stamp the dialled id onto a Chat Completions JSON answer. */
export function withDialledModel(answer: Bag, modelId: string): Bag {
  return { ...answer, model: modelId };
}

// --- streaming ---------------------------------------------------------------

export function sseFrame(eventName: string | null, payload: unknown): string {
  const head = eventName === null ? '' : `event: ${eventName}\n`;
  return `${head}data: ${JSON.stringify(payload)}\n\n`;
}

/** Rewrite the model field inside one raw SSE frame, keeping its shape. */
function restampFrame(frame: string, modelId: string): string {
  const lines = frame.split('\n').map((line) => {
    if (!line.startsWith('data:')) return line;
    const payload = line.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isBag(chunk)) return `data: ${JSON.stringify({ ...chunk, model: modelId })}`;
    } catch {
      // Not JSON — relay untouched.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** Relay an upstream Chat Completions SSE stream, restamping every chunk. */
export async function* relayChatStream(upstream: ReadableStream<Uint8Array>, modelId: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let cut = pending.indexOf('\n\n');
      while (cut !== -1) {
        const frame = pending.slice(0, cut);
        pending = pending.slice(cut + 2);
        yield restampFrame(frame, modelId);
        cut = pending.indexOf('\n\n');
      }
    }
    pending += decoder.decode();
    if (pending.trim() !== '') yield restampFrame(pending, modelId);
  } finally {
    reader.releaseLock();
  }
}

/**
 * Translate an upstream Chat Completions SSE stream into the Anthropic
 * event sequence: message_start, one content block per text run or tool
 * call (start / delta / stop), message_delta with the final stop reason
 * and output usage, message_stop. The finish reason is held until the
 * trailing usage chunk has been seen, because Anthropic's message_delta
 * is where output tokens belong.
 */
export async function* relayAsAnthropicEvents(
  upstream: ReadableStream<Uint8Array>,
  modelId: string,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  let opened = false;
  let closed = false;
  let heldStop: string | null = null;
  let textBlock = -1;
  let nextBlock = 0;
  const toolBlocks = new Map<number, number>();
  let inTokens = 0;
  let outTokens = 0;

  const opening = (): string[] => {
    if (opened) return [];
    opened = true;
    return [sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stickyroute_stream',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inTokens, output_tokens: 0 },
      },
    })];
  };

  const closings = (): string[] => {
    const out: string[] = [];
    if (textBlock !== -1) {
      out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: textBlock }));
      textBlock = -1;
    }
    const indexes = [...toolBlocks.values()].sort((a, b) => a - b);
    for (const index of indexes) out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index }));
    toolBlocks.clear();
    return out;
  };

  const finish = (stopReason: string): string[] => {
    if (closed) return [];
    closed = true;
    return [
      ...closings(),
      sseFrame('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outTokens },
      }),
      sseFrame('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsForFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...opening(), ...finish(heldStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isBag(chunk)) continue;
      if (isBag(chunk.usage)) {
        inTokens = num(chunk.usage.prompt_tokens);
        outTokens = num(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isBag(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isBag(choice.delta) ? choice.delta : {};
      out.push(...opening());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textBlock === -1) {
          textBlock = nextBlock;
          nextBlock += 1;
          out.push(sseFrame('content_block_start', {
            type: 'content_block_start',
            index: textBlock,
            content_block: { type: 'text', text: '' },
          }));
        }
        out.push(sseFrame('content_block_delta', {
          type: 'content_block_delta',
          index: textBlock,
          delta: { type: 'text_delta', text: delta.content },
        }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isBag(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolBlocks.size;
        let blockIndex = toolBlocks.get(upstreamIndex);
        if (blockIndex === undefined) {
          if (textBlock !== -1) {
            out.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: textBlock }));
            textBlock = -1;
          }
          blockIndex = nextBlock;
          nextBlock += 1;
          toolBlocks.set(upstreamIndex, blockIndex);
          const fn = isBag(call.function) ? call.function : {};
          out.push(sseFrame('content_block_start', {
            type: 'content_block_start',
            index: blockIndex,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${blockIndex}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }));
        }
        const fn = isBag(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseFrame('content_block_delta', {
            type: 'content_block_delta',
            index: blockIndex,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        heldStop = mapStopReason(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let cut = pending.indexOf('\n\n');
      while (cut !== -1) {
        const frame = pending.slice(0, cut);
        pending = pending.slice(cut + 2);
        for (const event of eventsForFrame(frame)) yield event;
        cut = pending.indexOf('\n\n');
      }
    }
    pending += decoder.decode();
    if (pending.trim() !== '') {
      for (const event of eventsForFrame(pending)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!closed) {
    for (const event of opening()) yield event;
    for (const event of finish(heldStop ?? 'end_turn')) yield event;
  }
}
