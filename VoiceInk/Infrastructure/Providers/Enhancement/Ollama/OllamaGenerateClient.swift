import Foundation
import LLMkit

/// Ollama's `think` value: an on/off switch, or a named level such as `"high"` for gpt-oss.
enum OllamaThink: Equatable, Sendable {
    case enabled(Bool)
    case level(String)

    var jsonValue: Any {
        switch self {
        case .enabled(let enabled): return enabled
        case .level(let level): return level
        }
    }
}

/// One Ollama `/api/generate` request. Optional fields are left out of the body when `nil`.
struct OllamaGenerateRequest: Sendable {
    var model: String
    var prompt: String
    var system: String
    var temperature: Double?
    var think: OllamaThink?

    var jsonBody: [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": false,
        ]
        // Same as LLMkit: `options` is only sent when it has a value.
        if let temperature { body["options"] = ["temperature": temperature] }
        if let think { body["think"] = think.jsonValue }
        return body
    }
}

struct OllamaGenerateResult: Equatable, Sendable {
    /// The answer. Ollama returns reasoning separately, in `thinking`.
    let response: String
    let thinking: String?
}

/// Client for Ollama's native generate API (`POST {baseURL}/api/generate`), non-streaming.
///
/// LLMkit's `OllamaClient.generate` only takes `think` as a Bool, so VoiceInk uses this client
/// for requests with a named thinking level.
enum OllamaGenerateClient {
    static func generate(
        baseURL: URL,
        request: OllamaGenerateRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> OllamaGenerateResult {
        let urlRequest = try ChatHTTP.jsonRequest(
            url: baseURL.appendingPathComponent("api/generate"),
            body: request.jsonBody
        )
        // LLMkit's Ollama client retries 429 and 500/502/503/504, not every 5xx.
        let data = try await ChatHTTP.send(
            urlRequest,
            timeout: timeout,
            retryOnTimeout: retryOnTimeout,
            retryableStatusCodes: ChatHTTP.defaultRetryableStatusCodes
        )
        let decoded = try ChatHTTP.decode(ResponseBody.self, from: data)
        return OllamaGenerateResult(response: decoded.response, thinking: decoded.thinking)
    }

    private struct ResponseBody: Decodable {
        let response: String
        let thinking: String?
    }
}
