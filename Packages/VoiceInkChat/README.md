# VoiceInkChat

VoiceInk's own chat library: text in, text out, for AI enhancement, the Assistant and Auto-Learn. It will replace LLMkit's chat clients. LLMkit stays for speech-to-text.

It depends on Foundation only, and the app doesn't link it yet.

## What's in it

| File | What it does |
|---|---|
| `ChatTypes.swift` | `ChatClient` protocol, `ChatRequest`, `ChatResult`, `FinishReason`, `ChatUsage` |
| `ChatHTTP.swift` | Shared transport: one reused `URLSession`, LLMkit's retry and timeout rules, error mapping, SSE streaming |
| `ChatError.swift` | `ChatError`, which mirrors `LLMKitError`, and provider error-body parsing |
| `OpenAICompatibleChatClient.swift` | OpenAI, Groq, Cerebras, Mistral, Together AI, GMI Cloud, Baseten, Fireworks AI, Vercel AI Gateway, Custom |
| `AnthropicChatClient.swift` | Anthropic Messages |
| `GeminiChatClient.swift` | Gemini Interactions, with v1/v1beta selection |
| `OpenRouterChatClient.swift` | OpenRouter, with `reasoning` and `provider` routing preferences |
| `OllamaChatClient.swift` | Ollama native `/api/chat`; `think` as a Bool or a level |
| `BedrockChatClient.swift` | Bedrock Converse with a bearer API key and `additionalModelRequestFields` |
| `ServerSentEvents.swift` | `text/event-stream` parser |
| `ReasoningText.swift` | Strips inline `<think>` sections |

## Using it

```swift
let client = OpenAICompatibleChatClient(apiKey: key, provider: .openAI)
let result = try await client.complete(ChatRequest(
    model: "gpt-6-luna",
    system: systemPrompt,
    messages: [.user(transcript)],
    temperature: 0.3,          // nil leaves the field out
    reasoning: .disabled,      // or .effort("low"), .enabled, .budget(tokens:)
    timeout: 7,
    retryOnTimeout: true
))
if result.isTruncated { /* don't paste a cut-off answer */ }
```

- `result.text` is the final answer only. Thinking blocks, `reasoning_content`, Mistral `thinking` chunks and `<think>` sections never reach it.
- `extraBody` is merged last, so it can add or override any field. `.null` sends an explicit `null`.
- A reasoning setting a provider's API can't express throws `ChatError.unsupported` before anything is sent. The table on `ChatRequest.Reasoning` shows each provider's mapping.

## Transport behaviour

`ChatHTTP` keeps the retry and timeout rules of LLMkit's chat clients, so a switched call site fails the same way:

- up to three attempts;
- 1 s then 2 s backoff for 429, every 5xx and network errors;
- timeouts (and HTTP 408) retried immediately, only with `retryOnTimeout`;
- the timeout limits each whole attempt;
- cancellation passes through.

The one change: LLMkit makes a new ephemeral session per attempt, while `ChatHTTP.shared` reuses one session, so repeat requests skip the TCP and TLS handshake.

## Tests

```sh
swift test                      # offline; a stub URLProtocol stands in for the network
swift run voiceink-chat-smoke   # live; every provider with a key in the environment
```

The smoke tool reads keys from environment variables only. The list is at the top of `Sources/voiceink-chat-smoke/main.swift`.

## License

GPL-3.0, like VoiceInk (`LICENSE`).

The wire code for several clients and the SSE parser is ported from [teunlao/swift-ai-sdk](https://github.com/teunlao/swift-ai-sdk) at commit `35c334a4`, under the Apache License 2.0 (`LICENSE-APACHE-2.0`). Each ported file starts with a header naming its source files and saying how it was modified. Upstream has no NOTICE file.
