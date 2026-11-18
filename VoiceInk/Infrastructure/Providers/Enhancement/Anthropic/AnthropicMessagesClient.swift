import Foundation
import LLMkit

/// The reasoning part of an Anthropic Messages request, for a level the user picked.
struct AnthropicReasoningRequest: Equatable, Sendable {
    enum Thinking: Equatable, Sendable {
        /// `{"type": "adaptive", "display": "omitted"}`: Claude decides how much to think, and
        /// thinking blocks come back without text.
        case adaptive
        /// `{"type": "enabled", "budget_tokens": N}`, for models that only have extended thinking.
        case budget(Int)
        /// `{"type": "disabled"}`.
        case disabled
        /// `{"type": "between_tools"}`: Claude Sonnet 5.5's way to skip up-front thinking.
        case betweenTools
    }

    var thinking: Thinking?
    /// `output_config.effort`.
    var effort: String?
    var maxTokens: Int

    /// The fields this adds to the request body.
    var bodyFields: [String: Any] {
        var fields: [String: Any] = ["max_tokens": maxTokens]
        switch thinking {
        case .adaptive:
            fields["thinking"] = ["type": "adaptive", "display": "omitted"]
        case .budget(let tokens):
            fields["thinking"] = ["type": "enabled", "budget_tokens": tokens, "display": "omitted"]
        case .disabled:
            fields["thinking"] = ["type": "disabled"]
        case .betweenTools:
            fields["thinking"] = ["type": "between_tools"]
        case nil:
            break
        }
        if let effort {
            fields["output_config"] = ["effort": effort]
        }
        return fields
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

/// Minimal client for the Anthropic Messages API (`POST https://api.anthropic.com/v1/messages`),
/// used when the user picks a reasoning level. Default requests still go through LLMkit's
/// `AnthropicLLMClient`, which can't send `thinking` or `output_config`.
///
/// The request matches LLMkit's apart from the reasoning fields: same headers and API
/// version, the same system prompt handling, and no sampling parameters (Claude 5 models
/// reject non-default `temperature`, `top_p` and `top_k`).
enum AnthropicMessagesClient {
    static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    static let apiVersion = "2023-06-01"

    static func createMessage(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        reasoning: AnthropicReasoningRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> AnthropicMessageResult {
        let request = try makeRequest(
            apiKey: apiKey,
            model: model,
            messages: messages,
            systemPrompt: systemPrompt,
            reasoning: reasoning
        )
        let data = try await ReasoningChatHTTP.send(request, timeout: timeout, retryOnTimeout: retryOnTimeout)
        return try decodeResult(from: data)
    }

    static func makeRequest(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        reasoning: AnthropicReasoningRequest
    ) throws -> URLRequest {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.missingAPIKey
        }

        // Same as LLMkit: an explicit system prompt wins, otherwise system messages are joined.
        let systemMessages = messages.filter { $0.role == "system" }.map(\.content)
        let system = systemPrompt ?? (systemMessages.isEmpty ? nil : systemMessages.joined(separator: "\n"))

        var body: [String: Any] = [
            "model": model,
            "messages": messages.filter { $0.role != "system" }.map { ["role": $0.role, "content": $0.content] },
        ]
        if let system {
            body["system"] = system
        }
        body.merge(reasoning.bodyFields) { _, new in new }

        return try ReasoningChatHTTP.jsonRequest(
            url: endpoint,
            body: body,
            headers: ["x-api-key": apiKey, "anthropic-version": apiVersion]
        )
    }

    static func decodeResult(from data: Data) throws -> AnthropicMessageResult {
        let decoded: ResponseBody
        do {
            decoded = try JSONDecoder().decode(ResponseBody.self, from: data)
        } catch {
            throw LLMKitError.decodingError(error.localizedDescription)
        }
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
