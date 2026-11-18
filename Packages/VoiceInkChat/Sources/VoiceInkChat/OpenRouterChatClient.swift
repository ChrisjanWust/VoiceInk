import Foundation

/// A client for OpenRouter's chat completions API: the OpenAI-compatible body plus OpenRouter's
/// `reasoning` object and `provider` routing preferences.
///
/// Given the same model, temperature, reasoning and routing preferences, it sends the body and
/// headers LLMkit's `OpenRouterClient.chatCompletion` sends. Beyond LLMkit, it:
/// - accepts any field through ``ChatRequest/extraBody`` and any header through
///   ``ChatRequest/extraHeaders``;
/// - adds the `order`, `only`, `ignore`, `quantizations`, `data_collection` and `zdr` routing
///   preferences;
/// - reads chunked `content` (models routed to Mistral) and strips inline `<think>` sections.
///
/// Reasoning: an explicit ``Options/reasoning`` wins. Otherwise ``ChatRequest/reasoning`` maps to
/// `reasoning.effort`, `.enabled`, `.max_tokens`, or `enabled: false`, always with
/// `exclude: true` so reasoning text isn't returned.
public struct OpenRouterChatClient: ChatClient {
    public static let defaultEndpoint = URL(string: "https://openrouter.ai/api/v1/chat/completions")!

    /// OpenRouter's `reasoning` object.
    public struct Reasoning: Sendable, Equatable {
        public var effort: String?
        public var maxTokens: Int?
        public var enabled: Bool?
        /// `true` keeps reasoning text out of the response.
        public var exclude: Bool?

        public init(effort: String? = nil, maxTokens: Int? = nil, enabled: Bool? = nil, exclude: Bool? = nil) {
            self.effort = effort
            self.maxTokens = maxTokens
            self.enabled = enabled
            self.exclude = exclude
        }

        var json: JSONValue {
            var object: [String: JSONValue] = [:]
            if let effort { object["effort"] = .string(effort) }
            if let maxTokens { object["max_tokens"] = .number(Double(maxTokens)) }
            if let enabled { object["enabled"] = .bool(enabled) }
            if let exclude { object["exclude"] = .bool(exclude) }
            return .object(object)
        }
    }

    /// OpenRouter's `provider` routing preferences.
    public struct ProviderPreferences: Sendable, Equatable {
        public enum Sort: String, Sendable {
            case price
            case throughput
            case latency
        }

        /// p50/p75/p90/p99 thresholds, in seconds for latency and tokens/s for throughput.
        public struct Percentiles: Sendable, Equatable {
            public var p50: Double?
            public var p75: Double?
            public var p90: Double?
            public var p99: Double?

            public init(p50: Double? = nil, p75: Double? = nil, p90: Double? = nil, p99: Double? = nil) {
                self.p50 = p50
                self.p75 = p75
                self.p90 = p90
                self.p99 = p99
            }

            var json: JSONValue {
                var object: [String: JSONValue] = [:]
                if let p50 { object["p50"] = .number(p50) }
                if let p75 { object["p75"] = .number(p75) }
                if let p90 { object["p90"] = .number(p90) }
                if let p99 { object["p99"] = .number(p99) }
                return .object(object)
            }
        }

        public var sort: Sort?
        public var preferredMaxLatency: Percentiles?
        public var preferredMinThroughput: Percentiles?
        public var requireParameters: Bool?
        public var allowFallbacks: Bool?
        /// Provider slugs to try first, in order.
        public var order: [String]?
        /// Provider slugs to allow; every other provider is skipped.
        public var only: [String]?
        public var ignore: [String]?
        public var quantizations: [String]?
        /// `"allow"` or `"deny"`.
        public var dataCollection: String?
        /// Only route to zero-data-retention endpoints.
        public var zdr: Bool?

        public init(
            sort: Sort? = nil,
            preferredMaxLatency: Percentiles? = nil,
            preferredMinThroughput: Percentiles? = nil,
            requireParameters: Bool? = nil,
            allowFallbacks: Bool? = nil,
            order: [String]? = nil,
            only: [String]? = nil,
            ignore: [String]? = nil,
            quantizations: [String]? = nil,
            dataCollection: String? = nil,
            zdr: Bool? = nil
        ) {
            self.sort = sort
            self.preferredMaxLatency = preferredMaxLatency
            self.preferredMinThroughput = preferredMinThroughput
            self.requireParameters = requireParameters
            self.allowFallbacks = allowFallbacks
            self.order = order
            self.only = only
            self.ignore = ignore
            self.quantizations = quantizations
            self.dataCollection = dataCollection
            self.zdr = zdr
        }

        var json: JSONValue {
            func strings(_ values: [String]) -> JSONValue { .array(values.map(JSONValue.string)) }
            var object: [String: JSONValue] = [:]
            if let sort { object["sort"] = .string(sort.rawValue) }
            if let preferredMaxLatency { object["preferred_max_latency"] = preferredMaxLatency.json }
            if let preferredMinThroughput { object["preferred_min_throughput"] = preferredMinThroughput.json }
            if let requireParameters { object["require_parameters"] = .bool(requireParameters) }
            if let allowFallbacks { object["allow_fallbacks"] = .bool(allowFallbacks) }
            if let order { object["order"] = strings(order) }
            if let only { object["only"] = strings(only) }
            if let ignore { object["ignore"] = strings(ignore) }
            if let quantizations { object["quantizations"] = strings(quantizations) }
            if let dataCollection { object["data_collection"] = .string(dataCollection) }
            if let zdr { object["zdr"] = .bool(zdr) }
            return .object(object)
        }
    }

    /// Per-request OpenRouter settings.
    public struct Options: Sendable, Equatable {
        /// Overrides the mapping of ``ChatRequest/reasoning``.
        public var reasoning: Reasoning?
        public var provider: ProviderPreferences?

        public init(reasoning: Reasoning? = nil, provider: ProviderPreferences? = nil) {
            self.reasoning = reasoning
            self.provider = provider
        }
    }

    public var apiKey: String
    /// Sends `X-OpenRouter-Metadata: enabled`, so the response says which provider served it.
    public var includeRouterMetadata: Bool
    /// Sent as `HTTP-Referer` for OpenRouter's app attribution.
    public var appReferer: URL?
    /// Sent as `X-OpenRouter-Title`.
    public var appTitle: String?
    /// Used by ``complete(_:)``, the protocol method. ``complete(_:options:)`` overrides it.
    public var defaultOptions: Options
    public var endpoint: URL
    public var http: ChatHTTP

    public init(
        apiKey: String,
        includeRouterMetadata: Bool = true,
        appReferer: URL? = nil,
        appTitle: String? = nil,
        defaultOptions: Options = Options(),
        endpoint: URL = defaultEndpoint,
        http: ChatHTTP = .shared
    ) {
        self.apiKey = apiKey
        self.includeRouterMetadata = includeRouterMetadata
        self.appReferer = appReferer
        self.appTitle = appTitle
        self.defaultOptions = defaultOptions
        self.endpoint = endpoint
        self.http = http
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await complete(request, options: defaultOptions)
    }

    public func complete(_ request: ChatRequest, options: Options) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request, options: options)
        let data = try await http.send(urlRequest, timeout: request.timeout, retryOnTimeout: request.retryOnTimeout)
        return try Self.decodeResult(from: data)
    }

    public func makeURLRequest(_ request: ChatRequest, options: Options) throws -> URLRequest {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        var headers = ["Authorization": "Bearer \(key)"]
        if includeRouterMetadata { headers["X-OpenRouter-Metadata"] = "enabled" }
        if let appReferer { headers["HTTP-Referer"] = appReferer.absoluteString }
        if let appTitle, !appTitle.isEmpty { headers["X-OpenRouter-Title"] = appTitle }
        headers.merge(request.extraHeaders) { _, new in new }
        return try ChatHTTP.jsonRequest(url: endpoint, body: try Self.body(for: request, options: options), headers: headers)
    }

    public static func body(for request: ChatRequest, options: Options) throws -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "model": .string(request.model),
            "messages": .array(OpenAICompatibleChatClient.messages(for: request)),
            "stream": false,
        ]
        if let temperature = request.temperature { body["temperature"] = .number(temperature) }
        if let maxTokens = request.maxTokens { body["max_tokens"] = .number(Double(maxTokens)) }
        if let serviceTier = request.serviceTier { body["service_tier"] = .string(serviceTier) }
        if let reasoning = options.reasoning ?? reasoning(for: request.reasoning) {
            body["reasoning"] = reasoning.json
        }
        if let provider = options.provider { body["provider"] = provider.json }
        return body.merging(extra: request.extraBody)
    }

    static func reasoning(for setting: ChatRequest.Reasoning?) -> Reasoning? {
        switch setting {
        case nil: return nil
        case .effort(let effort): return Reasoning(effort: effort, exclude: true)
        case .enabled: return Reasoning(enabled: true, exclude: true)
        case .disabled: return Reasoning(enabled: false, exclude: true)
        case .budget(let tokens): return Reasoning(maxTokens: tokens, exclude: true)
        }
    }

    public static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(OpenAIChatResponse.self, from: data)
        guard let choice = response.choices.first else {
            throw ChatError.noResultReturned
        }
        return try ChatResult.make(
            text: ReasoningText.strip(choice.message?.content?.answer ?? ""),
            finishReason: OpenAICompatibleChatClient.mapFinishReason(choice.finishReason),
            rawFinishReason: choice.finishReason,
            usage: OpenAIUsage.chatUsage(from: response.usage),
            model: response.model,
            provider: response.provider?.stringValue,
            metadata: response.openRouterMetadata
        )
    }
}
