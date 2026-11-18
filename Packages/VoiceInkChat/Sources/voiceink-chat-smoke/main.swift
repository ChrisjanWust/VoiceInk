// voiceink-chat-smoke: a live check of every VoiceInkChat client.
//
// It sends a dictation cleanup request, like VoiceInk's enhancement, to each provider that has a
// key in the environment, once at Default and once at a reasoning level, and prints the status,
// the clean answer and the latency. It's a developer tool; the app doesn't link it.
//
//   swift run voiceink-chat-smoke                 # every provider with a key
//   swift run voiceink-chat-smoke gemini bedrock  # only these providers
//
// Keys are read from environment variables only, and never printed:
//   OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY,
//   MISTRAL_API_KEY, TOGETHER_API_KEY, GMI_API_KEY, BASETEN_API_KEY, FIREWORKS_API_KEY,
//   VERCEL_AI_GATEWAY_API_KEY (or AI_GATEWAY_API_KEY), OPENROUTER_API_KEY,
//   AWS_BEARER_TOKEN_BEDROCK with AWS_REGION (default us-east-1),
//   OLLAMA_MODEL (with OLLAMA_BASE_URL, default http://localhost:11434).
// Set SMOKE_<PROVIDER>_MODEL, such as SMOKE_OPENAI_MODEL, to try another model.

import Foundation
import VoiceInkChat

let environment = ProcessInfo.processInfo.environment

func env(_ names: String...) -> String? {
    for name in names {
        if let value = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
            return value
        }
    }
    return nil
}

let systemPrompt = """
    You clean up dictated text. Remove filler words, false starts and repetitions, fix punctuation \
    and capitalization, and keep the speaker's meaning and wording. Reply with the cleaned text \
    only, with no preamble, quotes or explanation.
    """
let transcript = """
    um so i wanted to say that the uh the quarterly review is is moved to thursday at three \
    and please bring the the updated numbers
    """

struct Case {
    let provider: String
    let model: String
    let label: String
    let client: any ChatClient
    let request: ChatRequest
}

func request(
    _ model: String,
    temperature: Double? = nil,
    reasoning: ChatRequest.Reasoning? = nil,
    extraBody: [String: JSONValue] = [:],
    timeout: TimeInterval = 30
) -> ChatRequest {
    ChatRequest(
        model: model,
        system: systemPrompt,
        messages: [.user(transcript)],
        temperature: temperature,
        reasoning: reasoning,
        extraBody: extraBody,
        timeout: timeout,
        retryOnTimeout: false
    )
}

/// Default (VoiceInk's low-latency request for the model) and one picked level.
func pair(
    _ provider: String,
    _ client: any ChatClient,
    model: String,
    defaultRequest: ChatRequest,
    levelLabel: String,
    levelRequest: ChatRequest
) -> [Case] {
    [
        Case(provider: provider, model: model, label: "Default", client: client, request: defaultRequest),
        Case(provider: provider, model: model, label: levelLabel, client: client, request: levelRequest),
    ]
}

func openAICompatibleCases(
    _ key: String,
    _ provider: OpenAICompatibleProvider,
    id: String,
    model: String,
    defaultRequest: ChatRequest? = nil,
    level: String = "low"
) -> [Case] {
    let model = env("SMOKE_\(id.uppercased())_MODEL") ?? model
    let client = OpenAICompatibleChatClient(apiKey: key, provider: provider)
    return pair(
        id, client, model: model,
        defaultRequest: defaultRequest.map { var r = $0; r.model = model; return r } ?? request(model, temperature: 0.3),
        levelLabel: "effort \(level)",
        levelRequest: request(model, reasoning: .effort(level), timeout: 90)
    )
}

var cases: [Case] = []

if let key = env("OPENAI_API_KEY") {
    // VoiceInk's Default for GPT-6 Luna: no reasoning, temperature 0.3.
    cases += openAICompatibleCases(
        key, .openAI, id: "openai", model: "gpt-6-luna",
        defaultRequest: request("", temperature: 0.3, reasoning: .disabled)
    )
}
if let key = env("CEREBRAS_API_KEY") {
    // VoiceInk's Default for gpt-oss on Cerebras: lowest effort, reasoning hidden.
    cases += openAICompatibleCases(
        key, .cerebras, id: "cerebras", model: "gpt-oss-120b",
        defaultRequest: request("", temperature: 0.3, reasoning: .effort("low"), extraBody: ["reasoning_format": "hidden"]),
        level: "medium"
    )
}
if let key = env("GROQ_API_KEY") {
    cases += openAICompatibleCases(
        key, .groq, id: "groq", model: "openai/gpt-oss-20b",
        defaultRequest: request("", temperature: 0.3, reasoning: .effort("low"), extraBody: ["include_reasoning": false])
    )
}
if let key = env("MISTRAL_API_KEY") {
    cases += openAICompatibleCases(key, .mistral, id: "mistral", model: "mistral-small-latest", level: "high")
}
if let key = env("TOGETHER_API_KEY") {
    cases += openAICompatibleCases(key, .together, id: "together", model: "openai/gpt-oss-120b")
}
if let key = env("GMI_API_KEY") {
    cases += openAICompatibleCases(key, .gmiCloud, id: "gmi", model: "deepseek-ai/DeepSeek-V4.1-Flash")
}
if let key = env("BASETEN_API_KEY") {
    cases += openAICompatibleCases(key, .baseten, id: "baseten", model: "openai/gpt-oss-120b")
}
if let key = env("FIREWORKS_API_KEY") {
    cases += openAICompatibleCases(key, .fireworks, id: "fireworks", model: "accounts/fireworks/models/gpt-oss-120b")
}
if let key = env("VERCEL_AI_GATEWAY_API_KEY", "AI_GATEWAY_API_KEY") {
    cases += openAICompatibleCases(key, .vercelAIGateway, id: "vercel", model: "openai/gpt-oss-120b")
}
if let key = env("ANTHROPIC_API_KEY") {
    let model = env("SMOKE_ANTHROPIC_MODEL") ?? "claude-haiku-5-5"
    cases += pair(
        "anthropic", AnthropicChatClient(apiKey: key), model: model,
        defaultRequest: request(model),
        levelLabel: "effort low",
        levelRequest: request(model, reasoning: .effort("low"), timeout: 90)
    )
}
if let key = env("GEMINI_API_KEY") {
    // The app picks the version with GeminiAPIVersion.forModel and its v1 model list; these
    // are the versions that list serves today.
    let model = env("SMOKE_GEMINI_MODEL") ?? "gemini-3.6-flash"
    let v1 = GeminiChatClient(apiKey: key, apiVersion: .v1)
    cases += pair(
        "gemini", v1, model: model,
        defaultRequest: request(model, reasoning: .effort("minimal")),
        levelLabel: "thinking low",
        levelRequest: request(model, reasoning: .effort("low"), timeout: 90)
    )
    // xhigh, which LLMkit's closed enum can't send.
    cases.append(Case(
        provider: "gemini", model: "gemini-4-hydrogen", label: "thinking xhigh", client: v1,
        request: request("gemini-4-hydrogen", reasoning: .effort("xhigh"), timeout: 120)))
    // A v1beta-only alias, which LLMkit's "-preview" rule sends to v1, where it 404s.
    cases.append(Case(
        provider: "gemini", model: "gemini-flash-latest", label: "Default on v1beta",
        client: GeminiChatClient(apiKey: key, apiVersion: .v1beta), request: request("gemini-flash-latest")))
}

/// OpenRouter with VoiceInk's low-latency routing preferences.
struct RoutedOpenRouter: ChatClient {
    let client: OpenRouterChatClient
    func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await client.complete(
            request,
            providerPreferences: .init(
                sort: .throughput, preferredMaxLatencyP90: 2.5, requireParameters: true, allowFallbacks: true)
        )
    }
}

if let key = env("OPENROUTER_API_KEY") {
    let model = env("SMOKE_OPENROUTER_MODEL") ?? "openai/gpt-oss-120b"
    let client = RoutedOpenRouter(client: OpenRouterChatClient(
        apiKey: key, appReferer: URL(string: "https://tryvoiceink.com"), appTitle: "VoiceInk"))
    cases += pair(
        "openrouter", client, model: model,
        defaultRequest: request(model, reasoning: .effort("low")),
        levelLabel: "effort medium",
        levelRequest: request(model, reasoning: .effort("medium"), timeout: 90)
    )
}

/// Bedrock with model-specific fields in additionalModelRequestFields.
struct BedrockWithFields: ChatClient {
    let client: BedrockChatClient
    let fields: [String: JSONValue]
    func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await client.complete(request, additionalModelRequestFields: fields)
    }
}

if let key = env("AWS_BEARER_TOKEN_BEDROCK") {
    let region = env("AWS_REGION", "AWS_DEFAULT_REGION") ?? "us-east-1"
    let client = BedrockChatClient(apiKey: key, region: region)
    // Each model's low-effort reasoning fields. Nova and Claude reject unknown fields, so a pass
    // there shows the field was read; gpt-oss ignores unknown fields.
    let models: [(String, [String: JSONValue])] = [
        ("global.amazon.nova-2-lite-v1:0", ["reasoningConfig": ["type": "enabled", "maxReasoningEffort": "low"]]),
        ("global.anthropic.claude-sonnet-5-5",
         ["thinking": ["type": "adaptive", "display": "omitted"], "output_config": ["effort": "low"]]),
        ("openai.gpt-oss-120b-1:0", ["reasoning_effort": "low"]),
    ]
    for (model, fields) in models {
        cases.append(Case(provider: "bedrock", model: model, label: "Default", client: client, request: request(model)))
        cases.append(Case(
            provider: "bedrock", model: model, label: "effort low",
            client: BedrockWithFields(client: client, fields: fields), request: request(model, timeout: 90)))
    }
}
if let model = env("OLLAMA_MODEL") {
    let baseURL = env("OLLAMA_BASE_URL").flatMap(URL.init(string:)) ?? URL(string: "http://localhost:11434")!
    let client = OllamaChatClient(baseURL: baseURL)
    cases += pair(
        "ollama", client, model: model,
        defaultRequest: request(model, temperature: 0.3, reasoning: .disabled),
        levelLabel: "think on",
        levelRequest: request(model, temperature: 0.3, reasoning: .enabled, timeout: 120)
    )
}

let filter = Set(CommandLine.arguments.dropFirst().map { $0.lowercased() })
if !filter.isEmpty {
    cases = cases.filter { filter.contains($0.provider) }
}
if cases.isEmpty {
    print("No provider keys found in the environment. See the comment at the top of main.swift.")
    exit(2)
}

func oneLine(_ text: String, limit: Int = 110) -> String {
    let flat = text.replacingOccurrences(of: "\n", with: " ⏎ ")
    return flat.count > limit ? String(flat.prefix(limit)) + "…" : flat
}

/// Signs that reasoning leaked into the answer.
func leaksReasoning(_ text: String) -> Bool {
    let lowered = text.lowercased()
    return ["<think", "</think", "<thinking", "<reasoning"].contains { lowered.contains($0) }
}

var failures = 0
for item in cases {
    let started = ContinuousClock.now
    let status: String
    let detail: String
    do {
        let result = try await item.client.complete(item.request)
        let leaked = leaksReasoning(result.text)
        if result.isTruncated || leaked || result.text.isEmpty {
            failures += 1
            status = leaked ? "LEAK" : result.isTruncated ? "TRUNC" : "EMPTY"
        } else {
            status = "ok"
        }
        var usage = ""
        if let u = result.usage {
            usage = " [in \(u.inputTokens.map(String.init) ?? "?") out \(u.outputTokens.map(String.init) ?? "?")"
                + (u.reasoningTokens.map { " reasoning \($0)" } ?? "") + "]"
        }
        detail = "\(result.rawFinishReason ?? "-")\(usage) \(oneLine(result.text))"
    } catch {
        failures += 1
        status = "FAIL"
        detail = oneLine(error.localizedDescription, limit: 220)
    }
    let elapsed = started.duration(to: .now)
    let milliseconds = Int(Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15)
    let name = "\(item.provider) \(item.model) · \(item.label)"
    print("\(status.padding(toLength: 5, withPad: " ", startingAt: 0)) \(String(format: "%6d ms", milliseconds))  \(name)")
    print("      \(detail)")
}

print("\n\(cases.count - failures)/\(cases.count) passed")
exit(failures == 0 ? 0 : 1)
