import Foundation

/// A client for Ollama's native chat API (`POST {baseURL}/api/chat`), non-streaming.
///
/// Reasoning maps to `think`, which Ollama takes as a Bool or, for models such as gpt-oss, a
/// level: `.disabled` → `think: false`, `.enabled` → `think: true`, `.effort(e)` → `think: "e"`.
///
/// Ollama returns thinking in `message.thinking`, which isn't read. Models without `think`
/// support may write `<think>` sections into `content`; those are stripped.
///
/// `temperature` and `maxTokens` go in `options` (`num_predict`), which is only sent when it has a
/// field, as in LLMkit. Retries use LLMkit's Ollama statuses: 429, 500, 502, 503 and 504.
public struct OllamaChatClient: ChatClient {
    public var baseURL: URL
    var http = ChatHTTP.shared

    public init(baseURL: URL) {
        self.baseURL = baseURL
    }

    var endpoint: URL { baseURL.appendingPathComponent("api/chat") }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        let data = try await http.send(
            try makeURLRequest(request),
            timeout: request.timeout,
            retryOnTimeout: request.retryOnTimeout,
            retryableStatusCodes: ChatHTTP.defaultRetryableStatusCodes
        )
        return try Self.decodeResult(from: data)
    }

    func makeURLRequest(_ request: ChatRequest) throws -> URLRequest {
        try ChatHTTP.jsonRequest(url: endpoint, body: body(for: request), headers: request.extraHeaders)
    }

    func body(for request: ChatRequest) -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "messages": .array(OpenAICompatibleChatClient.messages(for: request)),
            "stream": false,
        ]
        var options: [String: JSONValue] = [:]
        if let temperature = request.temperature { options["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { options["num_predict"] = .number(Double(maxTokens)) }
        if !options.isEmpty { body["options"] = .object(options) }
        switch request.reasoning {
        case nil: break
        case .disabled: body["think"] = false
        case .enabled: body["think"] = true
        case .effort(let level): body["think"] = .string(level)
        }
        return body.merging(extra: request.extraBody)
    }

    static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(ResponseBody.self, from: data)
        let finishReason: FinishReason
        switch response.doneReason {
        case "stop": finishReason = .stop
        case "length": finishReason = .length
        default: finishReason = .other
        }
        return try ChatResult.make(
            text: ReasoningText.strip(response.message?.content ?? ""),
            finishReason: finishReason,
            rawFinishReason: response.doneReason,
            usage: ChatUsage(inputTokens: response.promptEvalCount, outputTokens: response.evalCount)
        )
    }

    private struct ResponseBody: Decodable {
        /// `thinking` holds the reasoning and isn't decoded.
        struct Message: Decodable {
            let content: String?
        }
        let message: Message?
        let doneReason: String?
        let promptEvalCount: Int?
        let evalCount: Int?

        enum CodingKeys: String, CodingKey {
            case message
            case doneReason = "done_reason"
            case promptEvalCount = "prompt_eval_count"
            case evalCount = "eval_count"
        }
    }
}
