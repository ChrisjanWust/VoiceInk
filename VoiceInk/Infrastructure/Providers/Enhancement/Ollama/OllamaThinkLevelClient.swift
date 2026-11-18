import Foundation
import LLMkit

/// Sends an Ollama `/api/generate` request with a named thinking level, such as
/// `"think": "high"` for gpt-oss. LLMkit's `OllamaClient.generate` only takes `think` as a
/// Bool, so it's used for everything else.
///
/// The body is LLMkit's (`model`, `prompt`, `system`, `options`, `stream: false`) with `think`
/// as a string. Ollama returns the reasoning in a separate `thinking` field, and only
/// `response` is read.
enum OllamaThinkLevelClient {
    static func generate(
        baseURL: URL,
        model: String,
        prompt: String,
        systemPrompt: String,
        temperature: Double,
        thinkLevel: String,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> String {
        let request = try makeRequest(
            baseURL: baseURL,
            model: model,
            prompt: prompt,
            systemPrompt: systemPrompt,
            temperature: temperature,
            thinkLevel: thinkLevel
        )
        // LLMkit's Ollama client retries 429 and 500/502/503/504, not every 5xx.
        let data = try await ReasoningChatHTTP.send(
            request,
            timeout: timeout,
            retryOnTimeout: retryOnTimeout,
            retryableStatusCodes: [429, 500, 502, 503, 504]
        )
        do {
            return try JSONDecoder().decode(ResponseBody.self, from: data).response
        } catch {
            throw LLMKitError.decodingError(error.localizedDescription)
        }
    }

    static func makeRequest(
        baseURL: URL,
        model: String,
        prompt: String,
        systemPrompt: String,
        temperature: Double,
        thinkLevel: String
    ) throws -> URLRequest {
        try ReasoningChatHTTP.jsonRequest(
            url: baseURL.appendingPathComponent("api/generate"),
            body: [
                "model": model,
                "prompt": prompt,
                "system": systemPrompt,
                "options": ["temperature": temperature],
                "stream": false,
                "think": thinkLevel,
            ],
            headers: [:]
        )
    }

    private struct ResponseBody: Decodable {
        let response: String
    }
}
