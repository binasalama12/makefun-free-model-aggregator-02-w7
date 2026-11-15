# StickyRoute Aggregator

One small server that puts every free model from the trial brief behind a
single endpoint, on both client wires at once.

Point an OpenAI client at `POST /v1/chat/completions` or an Anthropic
client at `POST /v1/messages` (either one streams when `stream: true`),
ask for model `auto`, and StickyRoute picks a free model for you, dials
it through the gateway, and hands the answer back in the wire you used.
`GET /v1/models` lists exactly the free catalog it can serve, in the
shape of whichever auth header you sent.

## Why "sticky"?

Prompts fall into kinds — wire, streaming or not, JSON mode or not,
tools or not. The first model that serves a kind well is remembered for
that kind and gets tried first next time, which keeps repeat prompts on
a proven route instead of re-rolling the dice. The memory is fussy,
though: a sticky favourite only leads while its published rating is
still within a small band of the best eligible model, so loyalty never
beats quality by much, and only a successful answer can earn or refresh
the slot. Everything else is ordered by the brief's ratings.

When the gateway refuses a model — a 429, a 5xx, a connection that goes
nowhere — that model is parked for a few rounds and the next candidate
takes over. If a later candidate serves the prompt, the refusals it
stepped over are reported on the `x-makefun-upstream-failures` response
header. Cooldowns run on a logical round counter, not a clock, so the
server behaves identically on a replay.

## Layout

- `main.ts` — the HTTP surface, the trial lifecycle (register, poll
  `/work`, exit when done) and the failover loop.
- `lib/brief.ts` — reads the brief, builds the catalog document that
  `--list` prints and registration posts.
- `lib/router.ts` — eligibility, rating order, the sticky cache and
  per-model cooldowns.
- `lib/wire.ts` — translation between the Anthropic and OpenAI shapes,
  for JSON bodies and for both streaming dialects.

## Running

In a trial the harness sets `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL`,
`OPENAI_BASE_URL` and `OPENAI_API_KEY`; the server needs nothing else
(Node 22, standard library only).

```sh
node main.ts --list   # print the catalog JSON and exit
node main.ts          # serve until the env says the work is done
```

No API key lives in this repository: the gateway key arrives through
the environment at run time, and the catalog only names the variable
(`OPENROUTER_API_KEY`) an entrant pushes to the competition.
