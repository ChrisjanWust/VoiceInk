// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/GoogleProvider/GoogleInteractionsLanguageModel.swift (request body)
// - Sources/GoogleProvider/GoogleInteractionsPrompt.swift (input steps)
// - Sources/GoogleProvider/GoogleInteractionsResponse.swift (steps, status, usage)
// Modified for VoiceInkChat: text-only, no tools, agents, background polling or streaming,
// `store: false` as in LLMkit, LLMkit's final-text rule, and v1/v1beta selection.

import Foundation

/// The Gemini API version an Interactions request goes to.
public enum GeminiAPIVersion: String, Sendable {
    /// The stable API, for generally available models.
    case v1
    /// The beta API. Preview models, and some aliases such as `-latest`, are only served here.
    case v1beta

    /// The version for a model: v1beta for `-preview` models and `-latest` aliases, v1 for the rest.
    ///
    /// LLMkit's rule only sent `-preview` models to v1beta, so the v1beta-only `-latest` aliases
    /// went to v1, where they 404. v1beta also serves every v1 model, so a model this rule sends
    /// to v1beta by mistake still works.
    public static func forModel(_ model: String) -> GeminiAPIVersion {
        let id = model.lowercased()
        return id.contains("-preview") || id.hasSuffix("-latest") ? .v1beta : .v1
    }
}

/// A client for Google's Gemini Interactions API (`POST /{version}/interactions`).
///
/// A Default request is the body LLMkit's `GeminiLLMClient` sends: `model`, `input`,
/// `system_instruction`, `store: false`, and `generation_config` only when it has a field.
///
/// Reasoning: `.effort(e)` sets `generation_config.thinking_level` to `e` as written, so any
/// level the API accepts can be sent, including `"xhigh"`. Gemini 3 models can't turn thinking
/// off, so `.disabled` and `.enabled` throw ``ChatError/unsupported(_:)``. Use
/// `.effort("minimal")` (or `"low"` where minimal isn't supported) for the least thinking.
public struct GeminiChatClient: ChatClient {
    public var apiKey: String
    /// Pick it with ``GeminiAPIVersion/forModel(_:)``.
    public var apiVersion: GeminiAPIVersion
    var baseURL = URL(string: "https://generativelanguage.googleapis.com")!
    var http = ChatHTTP.shared

    public init(apiKey: String, apiVersion: GeminiAPIVersion) {
        self.apiKey = apiKey
        self.apiVersion = apiVersion
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await complete(request, version: apiVersion)
    }

    private func complete(_ request: ChatRequest, version: GeminiAPIVersion) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request, version: version)
        let data = try await http.send(urlRequest, timeout: request.timeout, retryOnTimeout: request.retryOnTimeout)
        return try Self.decodeResult(from: data)
    }

    func endpoint(for version: GeminiAPIVersion) -> URL {
        baseURL.appendingPathComponent(version.rawValue).appendingPathComponent("interactions")
    }

    func makeURLRequest(_ request: ChatRequest, version: GeminiAPIVersion) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        let headers = ChatHTTP.mergedHeaders(request.extraHeaders, ["x-goog-api-key": key])
        return try ChatHTTP.jsonRequest(url: endpoint(for: version), body: try body(for: request), headers: headers)
    }

    func body(for request: ChatRequest) throws -> [String: JSONValue] {
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
            "model": .string(request.model),
            "input": .array(
                turns.map { turn in
                    [
                        "type": turn.role == .assistant ? "model_output" : "user_input",
                        "content": [["type": "text", "text": .string(turn.content)]],
                    ]
                }
            ),
            // Stateless, as in LLMkit and VoiceInk: Google doesn't keep the interaction.
            "store": false,
        ]
        if let system = request.resolvedSystem?.trimmingCharacters(in: .whitespacesAndNewlines), !system.isEmpty {
            body["system_instruction"] = .string(system)
        }

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
        }
        if !generation.isEmpty { body["generation_config"] = .object(generation) }
        return body.merging(extra: request.extraBody)
    }

    static func decodeResult(from data: Data) throws -> ChatResult {
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
            }
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
        let steps: [Step]?
        let usage: JSONValue?
    }
}
