// StickyRoute Aggregator — one endpoint, every free model behind it.
//
// A client points at this server with either wire (OpenAI Chat
// Completions or Anthropic Messages, JSON or streaming). For each
// request naming model "auto" the server works out what the request
// needs, asks the StickyRouter for its candidate order — the model that
// already served this prompt kind well goes first while its rating
// stays in the top band, then the rest by rating — and dials through
// the trial gateway. A 429, a 5xx or a network failure moves on to the
// next candidate and parks the refused model in a cooldown; the trail
// of refusals rides back on the x-makefun-upstream-failures header
// whenever a later candidate serves the prompt.
//
// Trial protocol: listen on 0.0.0.0 at an OS-picked port (never 8080),
// POST the catalog to $MAKEFUN_ENV_URL/register, serve until
// GET $MAKEFUN_ENV_URL/work says done, then exit 0.
// `node main.ts --list` prints the catalog and exits before any of that.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { catalogFrom, dialableModels, isBag, loadBrief, type Bag, type FreeModel } from './lib/brief.ts';
import { StickyRouter, kindKey, needsFromAnthropicBody, needsFromOpenAIBody, type Needs } from './lib/router.ts';
import {
  anthropicErrorBody,
  anthropicList,
  openAIErrorBody,
  openAIList,
  relayAsAnthropicEvents,
  relayChatStream,
  toChatBody,
  toMessageBody,
  withDialledModel,
  type ListedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const BODY_LIMIT = 8 * 1024 * 1024;
const DIAL_TIMEOUT_MS = 25_000;
const DEFAULT_MAX_TOKENS = 1024;
const GATEWAY_MAX_TOKENS = 8192;

const envBase = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const gatewayKey = process.env.OPENAI_API_KEY ?? '';

const brief = loadBrief(briefPath);
const catalog = catalogFrom(brief);
const pool: FreeModel[] = dialableModels(brief);
const router = new StickyRouter(pool);

type Wire = 'openai' | 'anthropic';

interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

function note(message: string): void {
  process.stderr.write(`stickyroute: ${message}\n`);
}

function gatewayChatUrl(): string {
  if (gatewayBase === '') return '';
  return gatewayBase.endsWith('/v1') ? `${gatewayBase}/chat/completions` : `${gatewayBase}/v1/chat/completions`;
}

function listedModels(): ListedModel[] {
  return pool.map((m) => ({ id: m.dialId, vendor: m.vendor, label: m.label }));
}

// --- request plumbing ---------------------------------------------------------

async function readBody(req: IncomingMessage): Promise<Bag | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    total += buf.length;
    if (total > BODY_LIMIT) return null;
    chunks.push(buf);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isBag(value) ? value : null;
  } catch {
    return null;
  }
}

function failuresHeader(failures: Refusal[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendWireError(res: ServerResponse, wire: Wire, status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicErrorBody(type, message), extra);
  else sendJson(res, status, openAIErrorBody(message, type, null), extra);
}

// --- gateway dialling -----------------------------------------------------------

/** The gateway refuses calls with no output cap (or one over its cap): normalise. */
function withOutputCap(body: Bag): Bag {
  const out = { ...body };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens
    : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens
    : null;
  if (cap === null) out.max_tokens = DEFAULT_MAX_TOKENS;
  else if (cap > GATEWAY_MAX_TOKENS) {
    if (typeof out.max_tokens === 'number') out.max_tokens = GATEWAY_MAX_TOKENS;
    else out.max_completion_tokens = GATEWAY_MAX_TOKENS;
  }
  return out;
}

function upstreamBody(wire: Wire, body: Bag, modelId: string): Bag {
  if (wire === 'anthropic') return withOutputCap(toChatBody(body, modelId));
  const out: Bag = { ...body, model: modelId };
  if (out.stream === true) {
    const opts = isBag(out.stream_options) ? out.stream_options : {};
    out.stream_options = { ...opts, include_usage: true };
  }
  return withOutputCap(out);
}

async function dialGateway(body: Bag): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${gatewayKey}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(gatewayChatUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/** Best-effort error code out of a gateway refusal body. */
function refusalCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isBag(parsed) && isBag(parsed.error)) {
      const code = parsed.error.code ?? parsed.error.type;
      if (typeof code === 'string' && code !== '') return code;
      if (typeof code === 'number') return String(code);
    }
  } catch {
    // Not JSON — fall through to the status-based label.
  }
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'upstream_error';
  return 'upstream_refusal';
}

function retryable(status: number): boolean {
  return status === 429 || status >= 500;
}

// --- serving one prompt -----------------------------------------------------------

async function serve(wire: Wire, body: Bag, res: ServerResponse): Promise<void> {
  router.tick();
  const needs: Needs = wire === 'anthropic' ? needsFromAnthropicBody(body) : needsFromOpenAIBody(body);
  const kind = kindKey(wire, body.stream === true, needs);
  const candidates = router.orderFor(kind, needs);
  if (candidates.length === 0) {
    sendWireError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model in the brief meets this request');
    return;
  }

  const failures: Refusal[] = [];
  let lastStatus = 502;
  let lastText = '';

  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dialGateway(upstreamBody(wire, body, candidate.dialId));
    } catch (err) {
      failures.push({ status: 0, code: 'network_error', model: candidate.dialId });
      router.noteRefusal(candidate.dialId, 0);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: candidate.dialId });
      lastStatus = upstream.status;
      lastText = text;
      if (retryable(upstream.status)) {
        router.noteRefusal(candidate.dialId, upstream.status);
        continue;
      }
      break; // A firm 4xx (bad request, unknown model): another model will not fix it.
    }

    // Served. Only now does the sticky cache learn anything.
    router.noteSuccess(kind, candidate.dialId);
    const extra = failuresHeader(failures);
    if (body.stream === true) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...extra,
      });
      try {
        if (wire === 'anthropic') {
          for await (const event of relayAsAnthropicEvents(upstream.body, candidate.dialId)) res.write(event);
        } else {
          for await (const frame of relayChatStream(upstream.body, candidate.dialId)) res.write(frame);
        }
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    const answerBag = isBag(answer) ? answer : {};
    if (wire === 'anthropic') sendJson(res, 200, toMessageBody(answerBag, candidate.dialId), extra);
    else sendJson(res, 200, withDialledModel(answerBag, candidate.dialId), extra);
    return;
  }

  const detail = lastText === '' ? '' : `: ${lastText.slice(0, 400)}`;
  sendWireError(res, wire, 502,
    wire === 'anthropic' ? 'api_error' : 'upstream_error',
    `every candidate failed; last gateway answer HTTP ${lastStatus}${detail}`,
    failuresHeader(failures));
}

// --- HTTP surface -------------------------------------------------------------------

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicList(models));
  else sendJson(res, 200, openAIList(models));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://stickyroute');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readBody(req);
    if (body === null) {
      sendWireError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readBody(req);
    if (body === null) {
      sendWireError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
      return;
    }
    await serve('anthropic', body, res);
    return;
  }
  sendWireError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`);
}

// --- trial lifecycle -------------------------------------------------------------------

function listenOnFreePort(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('server has no port'));
      else resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${envBase}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const reply: unknown = await res.json().catch(() => null);
  if (isBag(reply) && typeof reply.requests === 'number') note(`registered; trial holds ${reply.requests} prompts`);
}

function pollUntilDone(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${envBase}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json().catch(() => null);
          if (isBag(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // Env not reachable yet — next tick tries again.
      } finally {
        busy = false;
      }
    })();
  }, POLL_MS);
}

async function boot(): Promise<void> {
  if (brief.problem !== null) note(brief.problem);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (envBase === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendWireError(res, req.url?.startsWith('/v1/messages') ? 'anthropic' : 'openai', 500,
          'internal_error', 'the server failed to answer');
      }
    });
  });
  const port = await listenOnFreePort(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns: restart');
  note(`listening on 0.0.0.0:${port} with ${pool.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollUntilDone(server);
}

try {
  await boot();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
