# OpenRouter APIs — Black Forest Labs: FLUX.3 Image (black-forest-labs/flux-3-image)

Guide for calling every confirmed OpenRouter API that serves `black-forest-labs/flux-3-image`.

Model page: https://openrouter.ai/black-forest-labs/flux-3-image
Create an API key: https://openrouter.ai/settings/keys

## Authentication

Send this header with every request:

- Authorization: Bearer $OPENROUTER_API_KEY

## Text / Chat Completions API

Generate text with `black-forest-labs/flux-3-image` through OpenRouter's Chat Completions API.

Docs: https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion

### Endpoint

POST https://openrouter.ai/api/v1/chat/completions

Headers:
- Content-Type: application/json

### Request fields (black-forest-labs/flux-3-image)

- model: string (required) — `"black-forest-labs/flux-3-image"`
- messages: array (required) — ordered conversation messages with `role` and `content`
- stream: boolean (optional) — return Server-Sent Events as tokens are generated
- seed: optional — accepted by this model; see the API reference for its value shape

The model-specific optional fields above come from this model's advertised capabilities.
`stream` and `provider` routing preferences are API controls accepted independently of that
model capability list.

### Response

Without `stream`, the response is a Chat Completions JSON object:

```json
{
  "id": "gen-abc123",
  "choices": [{ "message": { "role": "assistant", "content": "..." } }],
  "usage": { "prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36, "cost": 0.001 }
}
```

With `stream: true`, the response is `text/event-stream`: read each `data:` JSON chunk until
`data: [DONE]`. The final usage chunk carries token counts and cost when usage is requested.

### Examples

#### Chat completion

```bash
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "black-forest-labs/flux-3-image",
  "messages": [
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
}'
```

### Error differences

- 400 — malformed messages or a parameter this model does not support
- 502 — the upstream text generation failed

## Image API

Generate images with `black-forest-labs/flux-3-image` through OpenRouter's Image API.

Docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation
Model discovery API: https://openrouter.ai/api/v1/images/models
Endpoints (black-forest-labs/flux-3-image-20261001): https://openrouter.ai/api/v1/images/models/black-forest-labs/flux-3-image-20261001/endpoints

### Endpoint

POST https://openrouter.ai/api/v1/images

Headers:
- Content-Type: application/json

### Request fields (black-forest-labs/flux-3-image)

- model: string (required) — `"black-forest-labs/flux-3-image"`
- prompt: string (required) — text description of the desired image
- resolution: "768" | "1K" | "1.5K" | "2K" | "4K" (optional) — resolution tier; concrete pixel dimensions are derived per provider
- aspect_ratio: "21:9" | "2:1" | "16:9" | "3:2" | "7:5" | "4:3" | "5:4" | "1:1" | "4:5" | "3:4" | "5:7" | "2:3" | "9:16" | "1:2" | "9:21" | "auto" (optional) — aspect ratio of the generated image
- n: integer 1-1 (optional) — upper bound on the number of images to generate; providers may return fewer, and single-image providers reject n > 1
- input_references: array of up to 10 image references (optional) — reference images for image-to-image, as `{ "type": "image_url", "image_url": { "url": "…" } }` entries; the url is an https URL or a base64 data URL

These are the generation parameters this model accepts between its providers; an
unlisted value is rejected, and a listed one can still be refused by whichever provider
serves the call. `provider` (routing preferences) is accepted on every request.

### Response

```json
{
  "created": 1748372400,
  "data": [{ "b64_json": "<base64 image bytes>", "media_type": "image/png" }],
  "usage": { "prompt_tokens": 0, "completion_tokens": 4175, "total_tokens": 4175, "cost": 0.04 }
}
```

Base64-decode `data[i].b64_json` and write the bytes to a file; `media_type` gives the extension.
`usage.cost` is the USD charge for the call.

### Examples

#### Text to Image

```bash
curl -X POST https://openrouter.ai/api/v1/images \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "black-forest-labs/flux-3-image",
  "prompt": "Editorial architectural photograph of a contemporary neighborhood storefront at blue hour, glowing warmly against the dusk. Above the entrance, a warm-white neon sign in flowing hand-bent cursive reads exactly \"OpenRouter\", the only text in the scene. Large windows reveal a cozy, lived-in interior: wood shelves styled with books and ceramics, lush trailing plants, and soft pendant lighting. Potted plants and a bicycle rest by the entrance, and golden light spills across the wet pavement in gentle reflections. Straight-on composition, realistic materials, quiet street with no people.",
  "n": 1,
  "aspect_ratio": "16:9",
  "resolution": "2K"
}'
```

#### Text to Image 2

```bash
curl -X POST https://openrouter.ai/api/v1/images \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "black-forest-labs/flux-3-image",
  "prompt": "Editorial architectural photograph of a contemporary neighborhood storefront at blue hour, glowing warmly against the dusk. Above the entrance, a warm-white neon sign in flowing hand-bent cursive reads exactly \"OpenRouter\", the only text in the scene. Large windows reveal a cozy, lived-in interior: wood shelves styled with books and ceramics, lush trailing plants, and soft pendant lighting. Potted plants and a bicycle rest by the entrance, and golden light spills across the wet pavement in gentle reflections. Straight-on composition, realistic materials, quiet street with no people.",
  "n": 1,
  "aspect_ratio": "16:9",
  "resolution": "4K"
}'
```

#### Text to Image 3

```bash
curl -X POST https://openrouter.ai/api/v1/images \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "black-forest-labs/flux-3-image",
  "prompt": "Editorial architectural photograph of a contemporary neighborhood storefront at blue hour, glowing warmly against the dusk. Above the entrance, a warm-white neon sign in flowing hand-bent cursive reads exactly \"OpenRouter\", the only text in the scene. Large windows reveal a cozy, lived-in interior: wood shelves styled with books and ceramics, lush trailing plants, and soft pendant lighting. Potted plants and a bicycle rest by the entrance, and golden light spills across the wet pavement in gentle reflections. Straight-on composition, realistic materials, quiet street with no people.",
  "n": 1,
  "aspect_ratio": "16:9",
  "resolution": "2K"
}'
```

#### Text to Image 4

```bash
curl -X POST https://openrouter.ai/api/v1/images \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "black-forest-labs/flux-3-image",
  "prompt": "Editorial architectural photograph of a contemporary neighborhood storefront at blue hour, glowing warmly against the dusk. Above the entrance, a warm-white neon sign in flowing hand-bent cursive reads exactly \"OpenRouter\", the only text in the scene. Large windows reveal a cozy, lived-in interior: wood shelves styled with books and ceramics, lush trailing plants, and soft pendant lighting. Potted plants and a bicycle rest by the entrance, and golden light spills across the wet pavement in gentle reflections. Straight-on composition, realistic materials, quiet street with no people.",
  "n": 1,
  "aspect_ratio": "16:9",
  "resolution": "2K"
}'
```

### Error differences

- 413 — request body too large

## Errors

Failures return `{"error": {"code": <number>, "message": <string>}}` with the HTTP status:

- 400 — malformed request or an unsupported parameter
- 401 — missing or invalid API key
- 402 — insufficient credits
- 403 — spend limit reached, key disabled, or access blocked
- 404 — unknown model or no provider can serve the request
- 429 — rate limited; retry with backoff
- 502 — the operation failed upstream; failed generations are not billed

---

Canonical version of this document: https://openrouter.ai/black-forest-labs/flux-3-image/llms.txt
