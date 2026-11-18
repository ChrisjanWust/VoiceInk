// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/AnthropicProvider/AnthropicMessagesAPI.swift
// - Sources/AnthropicProvider/MapAnthropicStopReason.swift
// Modified for VoiceInkChat: text-only messages, no tools, caching or streaming, LLMkit's
// system-prompt and max_tokens defaults, and the reasoning mapping below.

import Foundation

/// A client for the Anthropic Messages API (`POST /v1/messages`).
///
/// It sends LLMkit's headers and defaults, so a Default request is the same body LLMkit's
/// `AnthropicLLMClient` sends: `model`, `max_tokens` 8192, `system` and `messages`.
///
/// Reasoning:
/// - `.effort(e)`: `thinking: {type: adaptive, display: omitted}` and `output_config.effort`.
/// - `.enabled`: adaptive thinking at the model's default effort.
/// - `.disabled`: `thinking: {type: disabled}`.
///
/// For anything else, such as a token budget or `between_tools` thinking, set `thinking` (and
/// ``ChatRequest/maxTokens``) yourself through ``ChatRequest/extraBody``. A beta header goes in
/// ``ChatRequest/extraHeaders``.
/// Only `text` blocks reach the answer. `thinking` and `redacted_thinking` blocks are dropped.
public struct AnthropicChatClient: ChatClient {
    static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    static let apiVersion = "2023-06-01"
    /// LLMkit's default, used when ``ChatRequest/maxTokens`` is `nil`. The API requires the field.
    static let defaultMaxTokens = 8192

    public var apiKey: String
    var http = ChatHTTP.shared

    public init(apiKey: String) {
        self.apiKey = apiKey
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request)
        let data = try await http.send(urlRequest, timeout: request.timeout, retryOnTimeout: request.retryOnTimeout)
        return try Self.decodeResult(from: data)
    }

    func makeURLRequest(_ request: ChatRequest) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        let headers = ChatHTTP.mergedHeaders(
            request.extraHeaders,
            ["x-api-key": key, "anthropic-version": Self.apiVersion]
        )
        return try ChatHTTP.jsonRequest(url: Self.endpoint, body: try Self.body(for: request), headers: headers)
    }

    static func body(for request: ChatRequest) throws -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "max_tokens": .number(Double(request.maxTokens ?? defaultMaxTokens)),
            "messages": .array(
                request.conversation.map { ["role": .string($0.role.rawValue), "content": .string($0.content)] }
            ),
        ]
        if let system = request.resolvedSystem { body["system"] = .string(system) }
        if let temperature = request.temperature { body["temperature"] = .number(temperature) }

        switch request.reasoning {
        case nil:
            break
        case .effort(let effort):
            body["thinking"] = ["type": "adaptive", "display": "omitted"]
            body["output_config"] = ["effort": .string(effort)]
        case .enabled:
            body["thinking"] = ["type": "adaptive", "display": "omitted"]
        case .disabled:
            body["thinking"] = ["type": "disabled"]
        }
        return body.merging(extra: request.extraBody)
    }

    static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(ResponseBody.self, from: data)
        let text = response.content.filter { $0.type == "text" }.compactMap(\.text).joined()
        return try ChatResult.make(
            text: text,
            finishReason: mapStopReason(response.stopReason),
            rawFinishReason: response.stopReason,
            usage: response.usage.map {
                ChatUsage(inputTokens: $0["input_tokens"]?.intValue, outputTokens: $0["output_tokens"]?.intValue)
            }
        )
    }

    static func mapStopReason(_ reason: String?) -> FinishReason {
        switch reason {
        case "end_turn", "stop_sequence", "pause_turn": return .stop
        case "max_tokens", "model_context_window_exceeded": return .length
        case "refusal": return .contentFilter
        case "tool_use": return .toolCalls
        default: return .other
        }
    }

    private struct ResponseBody: Decodable {
        /// `thinking` blocks carry their text in `thinking` and `redacted_thinking` blocks in
        /// `data`. Neither field is decoded, so reasoning can't reach the answer.
        struct ContentBlock: Decodable {
            let type: String
            let text: String?
        }
        let content: [ContentBlock]
        let stopReason: String?
        let usage: JSONValue?

        enum CodingKeys: String, CodingKey {
            case content
            case stopReason = "stop_reason"
            case usage
        }
    }
}
