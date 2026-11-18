import Foundation

/// A client for Ollama's native chat API (`POST {baseURL}/api/chat`), non-streaming.
///
/// Reasoning maps to `think`, which Ollama takes as a Bool or, for models such as gpt-oss, a
/// level:
/// - `.disabled` → `think: false`, `.enabled` → `think: true`, `.effort(e)` → `think: "e"`.
/// - `.budget` throws ``ChatError/unsupported(_:)``.
///
/// Ollama returns thinking in `message.thinking`, which isn't read. Models without `think`
/// support may write `<think>` sections into `content`; those are stripped.
///
/// `temperature` and `maxTokens` go in `options` (`num_predict`), which is only sent when it has a
/// field, as in LLMkit. Retries use LLMkit's Ollama statuses: 429, 500, 502, 503 and 504.
public struct OllamaChatClient: ChatClient {
    public static let defaultBaseURL = URL(string: "http://localhost:11434")!

    public var baseURL: URL
    /// For Ollama behind an authenticating proxy, or Ollama's cloud API. Sent as a bearer token.
    public var apiKey: String?
    /// `keep_alive`, such as `"30m"`, which keeps the model loaded between requests.
    public var keepAlive: String?
    public var http: ChatHTTP

    public init(
        baseURL: URL = defaultBaseURL,
        apiKey: String? = nil,
        keepAlive: String? = nil,
        http: ChatHTTP = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.keepAlive = keepAlive
        self.http = http
    }

    public var endpoint: URL { baseURL.appendingPathComponent("api/chat") }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        let data = try await http.send(
            try makeURLRequest(request),
            timeout: request.timeout,
            retryOnTimeout: request.retryOnTimeout,
            retryableStatusCodes: ChatHTTP.defaultRetryableStatusCodes
        )
        return try Self.decodeResult(from: data)
    }

    public func makeURLRequest(_ request: ChatRequest) throws -> URLRequest {
        var headers = request.extraHeaders
        if let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty {
            headers["Authorization"] = "Bearer \(key)"
        }
        return try ChatHTTP.jsonRequest(url: endpoint, body: try body(for: request), headers: headers)
    }

    public func body(for request: ChatRequest) throws -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "messages": .array(OpenAICompatibleChatClient.messages(for: request)),
            "stream": false,
        ]
        var options: [String: JSONValue] = [:]
        if let temperature = request.temperature { options["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { options["num_predict"] = .number(Double(maxTokens)) }
        if !options.isEmpty { body["options"] = .object(options) }
        if let keepAlive { body["keep_alive"] = .string(keepAlive) }
        if request.serviceTier != nil {
            throw ChatError.unsupported("Ollama has no service tiers.")
        }
        switch request.reasoning {
        case nil: break
        case .disabled: body["think"] = false
        case .enabled: body["think"] = true
        case .effort(let level): body["think"] = .string(level)
        case .budget: throw ChatError.unsupported("Ollama's think takes true, false or a level, not a token budget.")
        }
        return body.merging(extra: request.extraBody)
    }

    public static func decodeResult(from data: Data) throws -> ChatResult {
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
            usage: ChatUsage(inputTokens: response.promptEvalCount, outputTokens: response.evalCount),
            model: response.model
        )
    }

    private struct ResponseBody: Decodable {
        /// `thinking` holds the reasoning and isn't decoded.
        struct Message: Decodable {
            let content: String?
        }

        let model: String?
        let message: Message?
        let doneReason: String?
        let promptEvalCount: Int?
        let evalCount: Int?

        enum CodingKeys: String, CodingKey {
            case model
            case message
            case doneReason = "done_reason"
            case promptEvalCount = "prompt_eval_count"
            case evalCount = "eval_count"
        }
    }
}
