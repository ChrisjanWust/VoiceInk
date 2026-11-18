import Foundation
import LLMkit

/// The `thinking` field of an Anthropic Messages request.
enum AnthropicThinking: Equatable, Sendable {
    /// How thinking blocks come back: with summary text, or with an empty `thinking` field.
    enum Display: String, Sendable {
        case summarized
        case omitted
    }

    /// `{"type": "adaptive"}`: Claude decides whether and how much to think.
    case adaptive(display: Display? = nil)
    /// `{"type": "enabled", "budget_tokens": N}`: extended thinking with a fixed budget.
    case enabled(budgetTokens: Int, display: Display? = nil)
    /// `{"type": "disabled"}`.
    case disabled
    /// `{"type": "between_tools"}`: no up-front thinking (Claude Sonnet 5.5).
    case betweenTools

    var jsonValue: [String: Any] {
        switch self {
        case .adaptive(let display):
            return ["type": "adaptive"].withDisplay(display)
        case .enabled(let budgetTokens, let display):
            return ["type": "enabled", "budget_tokens": budgetTokens].withDisplay(display)
        case .disabled:
            return ["type": "disabled"]
        case .betweenTools:
            return ["type": "between_tools"]
        }
    }
}

extension Dictionary where Key == String, Value == Any {
    fileprivate func withDisplay(_ display: AnthropicThinking.Display?) -> [String: Any] {
        guard let display else { return self }
        var copy = self
        copy["display"] = display.rawValue
        return copy
    }
}

/// One Anthropic Messages API request. Optional fields are left out of the body when `nil`.
struct AnthropicMessagesRequest: Sendable {
    var model: String
    var system: String?
    /// User and assistant turns, in order.
    var messages: [ChatMessage]
    /// LLMkit's default.
    var maxTokens: Int = 8192
    var thinking: AnthropicThinking?
    /// `output_config.effort`, such as `"low"` or `"max"`.
    var effort: String?
    /// Claude 5 models reject any non-default sampling parameter, so leave this `nil` for them.
    var temperature: Double?

    /// Builds a request the way LLMkit's `AnthropicLLMClient` does: an explicit system prompt
    /// wins, otherwise system messages are joined, and they're removed from `messages`.
    init(model: String, messages: [ChatMessage], systemPrompt: String?) {
        self.model = model
        let systemMessages = messages.filter { $0.role == "system" }.map(\.content)
        self.system = systemPrompt ?? (systemMessages.isEmpty ? nil : systemMessages.joined(separator: "\n"))
        self.messages = messages.filter { $0.role != "system" }
    }

    var jsonBody: [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
        ]
        if let system { body["system"] = system }
        if let thinking { body["thinking"] = thinking.jsonValue }
        if let effort { body["output_config"] = ["effort": effort] }
        if let temperature { body["temperature"] = temperature }
        return body
    }
}

struct AnthropicMessageResult: Equatable, Sendable {
    /// Every `text` block, joined. `thinking` and `redacted_thinking` blocks are never included.
    let text: String
    let stopReason: String?

    var wasTruncated: Bool {
        stopReason == "max_tokens" || stopReason == "model_context_window_exceeded"
    }
}

/// Client for the Anthropic Messages API (`POST https://api.anthropic.com/v1/messages`).
///
/// Sends LLMkit's headers and API version. VoiceInk uses it for requests LLMkit's
/// `AnthropicLLMClient` can't make, such as ones with `thinking` or `output_config`.
enum AnthropicMessagesClient {
    static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    static let apiVersion = "2023-06-01"

    static func createMessage(
        apiKey: String,
        request: AnthropicMessagesRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> AnthropicMessageResult {
        let urlRequest = try makeURLRequest(apiKey: apiKey, request: request)
        let data = try await ChatHTTP.send(urlRequest, timeout: timeout, retryOnTimeout: retryOnTimeout)
        return try decodeResult(from: data)
    }

    static func makeURLRequest(apiKey: String, request: AnthropicMessagesRequest) throws -> URLRequest {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.missingAPIKey
        }
        return try ChatHTTP.jsonRequest(
            url: endpoint,
            body: request.jsonBody,
            headers: ["x-api-key": apiKey, "anthropic-version": apiVersion]
        )
    }

    static func decodeResult(from data: Data) throws -> AnthropicMessageResult {
        let decoded = try ChatHTTP.decode(ResponseBody.self, from: data)
        let text = decoded.content
            .filter { $0.type == "text" }
            .compactMap(\.text)
            .joined()
        return AnthropicMessageResult(text: text, stopReason: decoded.stopReason)
    }

    private struct ResponseBody: Decodable {
        /// Only `text` blocks are read. `thinking` blocks carry their text in a `thinking` field
        /// and `redacted_thinking` blocks in `data`, so neither can leak into `text`.
        struct ContentBlock: Decodable {
            let type: String
            let text: String?
        }

        let content: [ContentBlock]
        let stopReason: String?

        enum CodingKeys: String, CodingKey {
            case content
            case stopReason = "stop_reason"
        }
    }
}
