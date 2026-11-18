// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/GoogleProvider/GoogleInteractionsLanguageModel.swift (request body)
// - Sources/GoogleProvider/GoogleInteractionsPrompt.swift (input steps)
// - Sources/GoogleProvider/GoogleInteractionsResponse.swift (steps, status, usage)
// Modified for VoiceInkChat: text-only, no tools, agents, background polling or streaming,
// `store: false` by default as in LLMkit, LLMkit's final-text rule, and v1/v1beta selection.

import Foundation

/// The Gemini API version an Interactions request goes to.
public enum GeminiAPIVersion: String, Sendable, Equatable, CaseIterable {
    /// The stable API, for generally available models.
    case v1
    /// The beta API. Preview models, and some aliases such as `-latest`, are only served here.
    case v1beta

    /// The version to use for a model, from the model lists each version serves
    /// (`GET /{version}/models`): v1 when it lists the model, otherwise v1beta.
    ///
    /// This replaces LLMkit's rule (a `-preview` substring means v1beta), which sends
    /// v1beta-only models such as the `-latest` aliases to v1, where they 404.
    public static func forModel(_ model: String, v1Models: Set<String>) -> GeminiAPIVersion {
        v1Models.contains(GeminiChatClient.bareModelID(model)) ? .v1 : .v1beta
    }
}

/// A client for Google's Gemini Interactions API (`POST /{version}/interactions`).
///
/// A Default request is the body LLMkit's `GeminiLLMClient` sends: `model`, `input`,
/// `system_instruction`, `store: false`, and `generation_config` only when it has a field.
///
/// Reasoning: `.effort(e)` sets `generation_config.thinking_level` to `e` as written, so any
/// level the API accepts can be sent, including `"xhigh"`. Gemini 3 models can't turn thinking
/// off and have no token budget, so `.disabled`, `.enabled` and `.budget` throw
/// ``ChatError/unsupported(_:)``. Use `.effort("minimal")` (or `"low"` where minimal isn't
/// supported) for the least thinking.
///
/// Endpoint: pass `apiVersion` when the caller knows it from the model lists (see
/// ``GeminiAPIVersion/forModel(_:v1Models:)``). With `nil`, the client tries v1 and, if v1 answers
/// 404, retries on v1beta and remembers that for the model for the rest of the process.
public struct GeminiChatClient: ChatClient {
    public static let baseURL = URL(string: "https://generativelanguage.googleapis.com")!

    public var apiKey: String
    public var apiVersion: GeminiAPIVersion?
    /// Whether Google keeps the interaction. LLMkit and VoiceInk send `false`.
    public var store: Bool
    public var baseURL: URL
    public var http: ChatHTTP

    public init(
        apiKey: String,
        apiVersion: GeminiAPIVersion? = nil,
        store: Bool = false,
        baseURL: URL = GeminiChatClient.baseURL,
        http: ChatHTTP = .shared
    ) {
        self.apiKey = apiKey
        self.apiVersion = apiVersion
        self.store = store
        self.baseURL = baseURL
        self.http = http
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        let model = Self.bareModelID(request.model)
        if let apiVersion {
            return try await complete(request, version: apiVersion)
        }
        if let remembered = await GeminiVersionMemory.shared.version(for: model) {
            return try await complete(request, version: remembered)
        }
        do {
            let result = try await complete(request, version: .v1)
            await GeminiVersionMemory.shared.remember(.v1, for: model)
            return result
        } catch ChatError.httpError(statusCode: 404, _, _) {
            let result = try await complete(request, version: .v1beta)
            await GeminiVersionMemory.shared.remember(.v1beta, for: model)
            return result
        }
    }

    private func complete(_ request: ChatRequest, version: GeminiAPIVersion) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request, version: version)
        let data = try await http.send(urlRequest, timeout: request.timeout, retryOnTimeout: request.retryOnTimeout)
        return try Self.decodeResult(from: data)
    }

    public func endpoint(for version: GeminiAPIVersion) -> URL {
        baseURL.appendingPathComponent(version.rawValue).appendingPathComponent("interactions")
    }

    public func makeURLRequest(_ request: ChatRequest, version: GeminiAPIVersion) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        let headers = request.extraHeaders.merging(["x-goog-api-key": key]) { _, auth in auth }
        return try ChatHTTP.jsonRequest(url: endpoint(for: version), body: try body(for: request), headers: headers)
    }

    public func body(for request: ChatRequest) throws -> [String: JSONValue] {
        // LLMkit's input rules: blank messages are dropped, there must be a turn left, and the
        // last one can't be the model's (Gemini doesn't continue a prefilled answer).
        let turns = request.conversation.filter {
            !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard let last = turns.last else {
            throw ChatError.unsupported("Gemini needs at least one non-empty user message.")
        }
        guard last.role != .assistant else {
            throw ChatError.unsupported("Gemini can't continue a prefilled assistant message.")
        }

        var body: [String: JSONValue] = [
            "model": .string(Self.bareModelID(request.model)),
            "input": .array(
                turns.map { turn in
                    [
                        "type": turn.role == .assistant ? "model_output" : "user_input",
                        "content": [["type": "text", "text": .string(turn.content)]],
                    ]
                }
            ),
            "store": .bool(store),
        ]
        if let system = request.resolvedSystem?.trimmingCharacters(in: .whitespacesAndNewlines), !system.isEmpty {
            body["system_instruction"] = .string(system)
        }
        if let serviceTier = request.serviceTier { body["service_tier"] = .string(serviceTier) }

        var generation: [String: JSONValue] = [:]
        if let temperature = request.temperature { generation["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { generation["max_output_tokens"] = .number(Double(maxTokens)) }
        switch request.reasoning {
        case nil:
            break
        case .effort(let level):
            generation["thinking_level"] = .string(level)
        case .disabled:
            throw ChatError.unsupported(
                "Gemini can't turn thinking off. Use .effort(\"minimal\") or the lowest level the model takes.")
        case .enabled:
            throw ChatError.unsupported("Gemini takes a thinking level. Use .effort(_:), or nil for the default.")
        case .budget:
            throw ChatError.unsupported("The Gemini Interactions API takes a thinking level, not a token budget.")
        }
        if !generation.isEmpty { body["generation_config"] = .object(generation) }
        return body.merging(extra: request.extraBody)
    }

    /// Accepts `models/gemini-…` as well as `gemini-…`, since the models endpoint lists names
    /// with the prefix.
    static func bareModelID(_ model: String) -> String {
        model.hasPrefix("models/") ? String(model.dropFirst("models/".count)) : model
    }

    public static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(ResponseBody.self, from: data)
        return try ChatResult.make(
            text: finalText(of: response.steps ?? []),
            finishReason: mapStatus(response.status),
            rawFinishReason: response.status,
            usage: response.usage.map {
                ChatUsage(
                    inputTokens: $0["total_input_tokens"]?.intValue,
                    outputTokens: $0["total_output_tokens"]?.intValue,
                    reasoningTokens: $0["total_thought_tokens"]?.intValue
                )
            },
            model: response.model
        )
    }

    /// LLMkit's rule: the text blocks at the end of the last `model_output` step that has text.
    /// `thought` steps hold the reasoning summary and are never read.
    static func finalText(of steps: [ResponseBody.Step]) -> String {
        for step in steps.reversed() where step.type == "model_output" {
            var blocks: [String] = []
            for block in (step.content ?? []).reversed() {
                if block.type == "text", let text = block.text {
                    blocks.insert(text, at: 0)
                } else if !blocks.isEmpty {
                    break
                }
            }
            let text = blocks.joined()
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return text }
        }
        return ""
    }

    static func mapStatus(_ status: String?) -> FinishReason {
        switch status {
        case "completed": return .stop
        case "incomplete": return .length
        case "failed", "cancelled": return .error
        case "requires_action": return .toolCalls
        default: return .other
        }
    }

    struct ResponseBody: Decodable {
        struct Step: Decodable {
            let type: String
            let content: [Block]?
        }

        struct Block: Decodable {
            let type: String
            let text: String?
        }

        let status: String?
        let model: String?
        let steps: [Step]?
        let usage: JSONValue?
    }
}

/// Which API version served each model, for clients with no explicit version.
actor GeminiVersionMemory {
    static let shared = GeminiVersionMemory()
    private var versions: [String: GeminiAPIVersion] = [:]

    func version(for model: String) -> GeminiAPIVersion? { versions[model] }
    func remember(_ version: GeminiAPIVersion, for model: String) { versions[model] = version }
    func reset() { versions = [:] }
}
