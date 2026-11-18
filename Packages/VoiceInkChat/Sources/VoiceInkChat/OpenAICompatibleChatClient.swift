// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/OpenAICompatibleProvider/Chat/OpenAICompatibleChatLanguageModel.swift
// - Sources/OpenAICompatibleProvider/Chat/MapOpenAICompatibleFinishReason.swift
// - Sources/MistralProvider/MistralChatLanguageModel.swift (chunked `content`)
// - Sources/MistralProvider/MapMistralFinishReason.swift
// Modified for VoiceInkChat: text-only messages, no tools or streaming deltas, optional
// temperature, per-provider presets, and reasoning kept out of the answer.

import Foundation

/// An OpenAI Chat Completions endpoint, with the differences between providers that share
/// the format.
public struct OpenAICompatibleProvider: Sendable, Equatable {
    /// A short name for logs and errors.
    public var name: String
    /// The full chat completions URL, such as `https://api.openai.com/v1/chat/completions`.
    public var endpoint: URL
    /// The body field for ``ChatRequest/maxTokens``. OpenAI deprecated `max_tokens` and its
    /// reasoning models reject it, so OpenAI, Groq and Cerebras use `max_completion_tokens`.
    public var maxTokensField: String
    /// Headers every request to this provider gets.
    public var headers: [String: String]

    public init(
        name: String,
        endpoint: URL,
        maxTokensField: String = "max_tokens",
        headers: [String: String] = [:]
    ) {
        self.name = name
        self.endpoint = endpoint
        self.maxTokensField = maxTokensField
        self.headers = headers
    }

    private static func preset(_ name: String, _ endpoint: String, maxTokensField: String = "max_tokens")
        -> OpenAICompatibleProvider
    {
        OpenAICompatibleProvider(name: name, endpoint: URL(string: endpoint)!, maxTokensField: maxTokensField)
    }

    public static let openAI = preset(
        "OpenAI", "https://api.openai.com/v1/chat/completions", maxTokensField: "max_completion_tokens")
    public static let groq = preset(
        "Groq", "https://api.groq.com/openai/v1/chat/completions", maxTokensField: "max_completion_tokens")
    public static let cerebras = preset(
        "Cerebras", "https://api.cerebras.ai/v1/chat/completions", maxTokensField: "max_completion_tokens")
    public static let mistral = preset("Mistral", "https://api.mistral.ai/v1/chat/completions")
    public static let together = preset("Together AI", "https://api.together.xyz/v1/chat/completions")
    public static let gmiCloud = preset("GMI Cloud", "https://api.gmi-serving.com/v1/chat/completions")
    public static let baseten = preset("Baseten", "https://inference.baseten.co/v1/chat/completions")
    public static let fireworks = preset("Fireworks AI", "https://api.fireworks.ai/inference/v1/chat/completions")
    public static let vercelAIGateway = preset("Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1/chat/completions")

    /// A user-configured endpoint. `endpoint` is the full chat completions URL, as VoiceInk's
    /// custom provider settings store it.
    public static func custom(endpoint: URL, name: String = "Custom") -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(name: name, endpoint: endpoint)
    }
}

/// A client for OpenAI Chat Completions and every provider that copies it: OpenAI, Groq,
/// Cerebras, Mistral, Together AI, GMI Cloud, Baseten, Fireworks AI, Vercel AI Gateway and
/// custom endpoints.
///
/// What it does differently from LLMkit's `OpenAILLMClient`:
/// - `temperature` is left out when it's `nil`, instead of always sending 0.3.
/// - `message.content` may be a string, `null`, or Mistral's list of `text` and `thinking`
///   chunks. Only answer text is kept.
/// - `reasoning_content`, `reasoning` and inline `<think>` sections never reach the answer.
/// - `finish_reason` and `usage` are returned.
public struct OpenAICompatibleChatClient: ChatClient {
    public var apiKey: String
    public var provider: OpenAICompatibleProvider
    public var http: ChatHTTP

    public init(apiKey: String, provider: OpenAICompatibleProvider, http: ChatHTTP = .shared) {
        self.apiKey = apiKey
        self.provider = provider
        self.http = http
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request)
        let data = try await http.send(
            urlRequest,
            timeout: request.timeout,
            retryOnTimeout: request.retryOnTimeout
        )
        return try Self.decodeResult(from: data)
    }

    public func makeURLRequest(_ request: ChatRequest) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        let headers = provider.headers
            .merging(["Authorization": "Bearer \(key)"]) { _, new in new }
            .merging(request.extraHeaders) { _, new in new }
        return try ChatHTTP.jsonRequest(url: provider.endpoint, body: try body(for: request), headers: headers)
    }

    /// The request body. `extraBody` is applied last, so it can override or `null` any field.
    public func body(for request: ChatRequest) throws -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "messages": .array(Self.messages(for: request)),
            "stream": false,
        ]
        if let temperature = request.temperature { body["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { body[provider.maxTokensField] = .number(Double(maxTokens)) }
        if let serviceTier = request.serviceTier { body["service_tier"] = .string(serviceTier) }
        switch request.reasoning {
        case nil:
            break
        case .effort(let effort):
            body["reasoning_effort"] = .string(effort)
        case .disabled:
            body["reasoning_effort"] = "none"
        case .enabled:
            throw ChatError.unsupported(
                "\(provider.name) takes a reasoning effort, not an on/off switch. Use .effort(_:).")
        case .budget:
            throw ChatError.unsupported("\(provider.name) takes a reasoning effort, not a token budget.")
        }
        return body.merging(extra: request.extraBody)
    }

    /// LLMkit's message order: a non-empty system prompt first, then every message as given.
    static func messages(for request: ChatRequest) -> [JSONValue] {
        var messages = request.messages
        if let system = request.system, !system.isEmpty {
            messages.insert(.system(system), at: 0)
        }
        return messages.map { ["role": .string($0.role.rawValue), "content": .string($0.content)] }
    }

    public static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(OpenAIChatResponse.self, from: data)
        guard let choice = response.choices.first else {
            throw ChatError.noResultReturned
        }
        return try ChatResult.make(
            text: ReasoningText.strip(choice.message?.content?.answer ?? ""),
            finishReason: mapFinishReason(choice.finishReason),
            rawFinishReason: choice.finishReason,
            usage: OpenAIUsage.chatUsage(from: response.usage),
            model: response.model
        )
    }

    /// OpenAI's finish reasons, plus Mistral's `model_length` and the `max_tokens` some
    /// gateways pass through from Anthropic models.
    static func mapFinishReason(_ reason: String?) -> FinishReason {
        switch reason?.lowercased() {
        case "stop", "end_turn", "stop_sequence", "eos": return .stop
        case "length", "model_length", "max_tokens", "max_output_tokens": return .length
        case "content_filter": return .contentFilter
        case "tool_calls", "function_call": return .toolCalls
        case "error": return .error
        default: return .other
        }
    }
}

// MARK: - Response body

/// The parts of a Chat Completions response VoiceInk reads. Reasoning fields
/// (`reasoning_content`, `reasoning`, `reasoning_details`) aren't decoded at all.
struct OpenAIChatResponse: Decodable {
    struct Choice: Decodable {
        let message: Message?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case message
            case finishReason = "finish_reason"
        }
    }

    struct Message: Decodable {
        let content: MessageContent?
    }

    let model: String?
    let provider: JSONValue?
    let choices: [Choice]
    let usage: JSONValue?
    let openRouterMetadata: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model
        case provider
        case choices
        case usage
        case openRouterMetadata = "openrouter_metadata"
    }
}

/// `message.content`: a string, `null`, or a list of chunks.
///
/// Mistral's reasoning models return `[{"type": "thinking", "thinking": [...]}, {"type": "text",
/// "text": "..."}]`. Only top-level `text` chunks are the answer. A `thinking` chunk nests its own
/// text under `thinking`, which isn't read, so reasoning can't leak into the answer.
enum MessageContent: Decodable {
    case text(String)
    case chunks([Chunk])

    struct Chunk: Decodable {
        let type: String?
        let text: String?
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .chunks(try container.decode([Chunk].self))
        }
    }

    var answer: String {
        switch self {
        case .text(let text):
            return text
        case .chunks(let chunks):
            return chunks.filter { $0.type == "text" || $0.type == "output_text" }.compactMap(\.text).joined()
        }
    }
}

/// Reads `usage` loosely, so an odd field in it can't fail the whole response.
enum OpenAIUsage {
    static func chatUsage(from usage: JSONValue?) -> ChatUsage? {
        guard let usage else { return nil }
        return ChatUsage(
            inputTokens: usage["prompt_tokens"]?.intValue,
            outputTokens: usage["completion_tokens"]?.intValue,
            reasoningTokens: usage["completion_tokens_details"]?["reasoning_tokens"]?.intValue
        )
    }
}
