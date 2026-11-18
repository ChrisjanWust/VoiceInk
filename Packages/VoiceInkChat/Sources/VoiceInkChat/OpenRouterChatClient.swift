import Foundation

/// A client for OpenRouter's chat completions API: the OpenAI-compatible body plus OpenRouter's
/// `reasoning` object and `provider` routing preferences.
///
/// Given the same model, temperature, reasoning and routing preferences, it sends the body and
/// headers LLMkit's `OpenRouterClient.chatCompletion` sends, including `X-OpenRouter-Metadata`
/// so the response says which provider served it. Beyond LLMkit, it reads chunked `content`
/// (models routed to Mistral) and strips inline `<think>` sections.
///
/// Reasoning maps to `reasoning`, always with `exclude: true` so reasoning text isn't returned,
/// as VoiceInk's `OpenRouterRequestPolicy` does: `.effort(e)` → `effort`, `.disabled` →
/// `enabled: false`, `.enabled` → `enabled: true`.
public struct OpenRouterChatClient: ChatClient {
    static let endpoint = URL(string: "https://openrouter.ai/api/v1/chat/completions")!

    /// OpenRouter's `provider` routing preferences: the fields VoiceInk's low-latency policy sets.
    public struct ProviderPreferences: Sendable, Equatable {
        public enum Sort: String, Sendable {
            case price
            case throughput
            case latency
        }

        public var sort: Sort?
        /// `preferred_max_latency.p90`, in seconds.
        public var preferredMaxLatencyP90: Double?
        public var requireParameters: Bool?
        public var allowFallbacks: Bool?

        public init(
            sort: Sort? = nil,
            preferredMaxLatencyP90: Double? = nil,
            requireParameters: Bool? = nil,
            allowFallbacks: Bool? = nil
        ) {
            self.sort = sort
            self.preferredMaxLatencyP90 = preferredMaxLatencyP90
            self.requireParameters = requireParameters
            self.allowFallbacks = allowFallbacks
        }

        var json: JSONValue {
            var object: [String: JSONValue] = [:]
            if let sort { object["sort"] = .string(sort.rawValue) }
            if let preferredMaxLatencyP90 { object["preferred_max_latency"] = ["p90": .number(preferredMaxLatencyP90)] }
            if let requireParameters { object["require_parameters"] = .bool(requireParameters) }
            if let allowFallbacks { object["allow_fallbacks"] = .bool(allowFallbacks) }
            return .object(object)
        }
    }

    public var apiKey: String
    /// Sent as `HTTP-Referer` for OpenRouter's app attribution.
    public var appReferer: URL?
    /// Sent as `X-OpenRouter-Title`.
    public var appTitle: String?
    var http = ChatHTTP.shared

    public init(apiKey: String, appReferer: URL? = nil, appTitle: String? = nil) {
        self.apiKey = apiKey
        self.appReferer = appReferer
        self.appTitle = appTitle
    }

    /// Sends the request with no routing preferences.
    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await complete(request, providerPreferences: nil)
    }

    public func complete(
        _ request: ChatRequest,
        providerPreferences: ProviderPreferences?
    ) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request, providerPreferences: providerPreferences)
        let data = try await http.send(urlRequest, timeout: request.timeout, retryOnTimeout: request.retryOnTimeout)
        return try Self.decodeResult(from: data)
    }

    func makeURLRequest(_ request: ChatRequest, providerPreferences: ProviderPreferences?) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        var headers = ["X-OpenRouter-Metadata": "enabled"]
        if let appReferer { headers["HTTP-Referer"] = appReferer.absoluteString }
        if let appTitle, !appTitle.isEmpty { headers["X-OpenRouter-Title"] = appTitle }
        headers = ChatHTTP.mergedHeaders(headers, request.extraHeaders, ["Authorization": "Bearer \(key)"])
        return try ChatHTTP.jsonRequest(
            url: Self.endpoint,
            body: Self.body(for: request, providerPreferences: providerPreferences),
            headers: headers
        )
    }

    static func body(for request: ChatRequest, providerPreferences: ProviderPreferences?) -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "messages": .array(OpenAICompatibleChatClient.messages(for: request)),
            "stream": false,
        ]
        if let temperature = request.temperature { body["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { body["max_tokens"] = .number(Double(maxTokens)) }
        switch request.reasoning {
        case nil: break
        case .effort(let effort): body["reasoning"] = ["effort": .string(effort), "exclude": true]
        case .disabled: body["reasoning"] = ["enabled": false, "exclude": true]
        case .enabled: body["reasoning"] = ["enabled": true, "exclude": true]
        }
        if let providerPreferences { body["provider"] = providerPreferences.json }
        return body.merging(extra: request.extraBody)
    }

    static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(OpenAIChatResponse.self, from: data)
        guard let choice = response.choices.first else {
            throw ChatError.noResultReturned
        }
        return try ChatResult.make(
            text: ReasoningText.strip(choice.message?.content?.answer ?? ""),
            finishReason: OpenAICompatibleChatClient.mapFinishReason(choice.finishReason),
            rawFinishReason: choice.finishReason,
            usage: OpenAIUsage.chatUsage(from: response.usage),
            provider: response.provider?.stringValue,
            metadata: response.openRouterMetadata
        )
    }
}
