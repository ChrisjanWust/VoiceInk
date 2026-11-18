// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/AmazonBedrockProvider/BedrockChatLanguageModel.swift (Converse body, per-family
//   reasoning fields in additionalModelRequestFields, serviceTier)
// - Sources/AmazonBedrockProvider/MapBedrockFinishReason.swift
// - Sources/AmazonBedrockProvider/BedrockProvider.swift (bearer API keys)
// Modified for VoiceInkChat: text-only, no tools, SigV4, guardrails or streaming; the message
// normalisation and trailing-space cleanup from VoiceInk's in-app BedrockConverseClient.

import Foundation

/// A client for the Amazon Bedrock Runtime Converse API, authenticated with a Bedrock API key
/// (a bearer token).
///
/// `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse`
///
/// Reasoning goes in `additionalModelRequestFields`, in the shape each model family expects:
/// - Anthropic (`anthropic.` IDs): the same `thinking` and `output_config` as
///   ``AnthropicChatClient``. A budget is added on top of `inferenceConfig.maxTokens`.
/// - OpenAI gpt-oss: `reasoning_effort`. Other OpenAI models: `reasoning.effort`.
/// - Amazon Nova 2: `reasoningConfig` with `type` and `maxReasoningEffort`.
/// - Other families have no known field, so a reasoning setting throws ``ChatError/unsupported(_:)``.
///
/// ``additionalModelRequestFields`` in ``complete(_:additionalModelRequestFields:)`` is merged
/// over that mapping, and ``ChatRequest/extraBody`` over the top-level Converse body.
/// ``ChatRequest/serviceTier`` becomes `serviceTier: {type: …}`.
///
/// Only `text` blocks reach the answer; `reasoningContent` blocks are dropped.
public struct BedrockChatClient: ChatClient {
    /// Statuses the in-app client retries: throttling and server errors, including 529.
    /// 424 (`ModelErrorException`) isn't retried, since the model itself failed.
    public static let retryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504, 529]

    public var apiKey: String
    public var region: String
    public var http: ChatHTTP

    public init(apiKey: String, region: String, http: ChatHTTP = .shared) {
        self.apiKey = apiKey
        self.region = region
        self.http = http
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await complete(request, additionalModelRequestFields: [:])
    }

    public func complete(
        _ request: ChatRequest,
        additionalModelRequestFields: [String: JSONValue]
    ) async throws -> ChatResult {
        let urlRequest = try makeURLRequest(request, additionalModelRequestFields: additionalModelRequestFields)
        let data = try await http.send(
            urlRequest,
            timeout: request.timeout,
            retryOnTimeout: request.retryOnTimeout,
            retryableStatusCodes: Self.retryableStatusCodes,
            errorTypeHeader: "x-amzn-ErrorType"
        )
        return try Self.decodeResult(from: data)
    }

    // MARK: - Request

    public static func normalizedRegion(_ region: String) -> String {
        region.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Accepts keys pasted with a `Bearer ` prefix.
    public static func normalizedAPIKey(_ apiKey: String) -> String {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let prefix = key.range(of: #"^bearer(\s+|$)"#, options: [.regularExpression, .caseInsensitive]) else {
            return key
        }
        return String(key[prefix.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Region codes such as `us-east-1`, `eu-central-2` and `us-gov-west-1`.
    public static func isValidRegion(_ region: String) -> Bool {
        normalizedRegion(region).range(of: #"^[a-z]{2}(-[a-z]+)+-[0-9]{1,2}$"#, options: .regularExpression) != nil
    }

    public static func endpoint(region: String, modelID: String) throws -> URL {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        let encodedModelID = modelID.addingPercentEncoding(withAllowedCharacters: allowed) ?? modelID
        let string = "https://bedrock-runtime.\(normalizedRegion(region)).amazonaws.com/model/\(encodedModelID)/converse"
        guard let url = URL(string: string) else { throw ChatError.invalidURL(string) }
        return url
    }

    public func makeURLRequest(
        _ request: ChatRequest,
        additionalModelRequestFields: [String: JSONValue] = [:]
    ) throws -> URLRequest {
        let key = Self.normalizedAPIKey(apiKey)
        guard !key.isEmpty else { throw ChatError.missingAPIKey }
        guard Self.isValidRegion(region) else {
            throw ChatError.invalidURL("\"\(region)\" isn't an AWS Region code, such as us-east-1.")
        }
        let modelID = request.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else { throw ChatError.unsupported("Bedrock needs a model ID or inference profile ID.") }

        var request = request
        request.model = modelID
        let headers = ["Authorization": "Bearer \(key)", "Accept": "application/json"]
            .merging(request.extraHeaders) { _, new in new }
        return try ChatHTTP.jsonRequest(
            url: try Self.endpoint(region: region, modelID: modelID),
            body: try Self.body(for: request, additionalModelRequestFields: additionalModelRequestFields),
            headers: headers
        )
    }

    public static func body(
        for request: ChatRequest,
        additionalModelRequestFields extra: [String: JSONValue] = [:]
    ) throws -> [String: JSONValue] {
        var body: [String: JSONValue] = [
            "messages": .array(
                normalizedTurns(request.conversation).map {
                    ["role": .string($0.role.rawValue), "content": [["text": .string($0.content)]]]
                }
            )
        ]
        if let system = request.resolvedSystem, !system.isEmpty {
            body["system"] = [["text": .string(system)]]
        }

        var inference: [String: JSONValue] = [:]
        if let maxTokens = request.maxTokens { inference["maxTokens"] = .number(Double(maxTokens)) }
        if let temperature = request.temperature { inference["temperature"] = .number(temperature) }

        var additional = try reasoningFields(for: request.reasoning, modelID: request.model)
        if case .budget(let tokens) = request.reasoning {
            // Same rule as the Anthropic client: the budget comes on top of the answer's tokens.
            inference["maxTokens"] = .number(Double((request.maxTokens ?? AnthropicChatClient.defaultMaxTokens) + tokens))
        }
        additional = additional.merging(extra: extra)

        if !inference.isEmpty { body["inferenceConfig"] = .object(inference) }
        if !additional.isEmpty { body["additionalModelRequestFields"] = .object(additional) }
        if let serviceTier = request.serviceTier { body["serviceTier"] = ["type": .string(serviceTier)] }
        return body.merging(extra: request.extraBody)
    }

    enum ModelFamily: Equatable {
        case anthropic
        case gptOSS
        case openAI
        case nova
        case other
    }

    /// The family from a model ID, inference profile ID (`us.anthropic.…`) or ARN.
    static func family(of modelID: String) -> ModelFamily {
        let id = modelID.lowercased()
        if id.contains("anthropic.") { return .anthropic }
        if id.contains("openai.gpt-oss") { return .gptOSS }
        if id.range(of: #"(^|[.:/])openai\."#, options: .regularExpression) != nil { return .openAI }
        if id.contains("amazon.nova-2") { return .nova }
        return .other
    }

    static func reasoningFields(for reasoning: ChatRequest.Reasoning?, modelID: String) throws -> [String: JSONValue] {
        guard let reasoning else { return [:] }
        let family = family(of: modelID)
        switch (family, reasoning) {
        case (.anthropic, .effort(let effort)):
            return ["thinking": ["type": "adaptive", "display": "omitted"], "output_config": ["effort": .string(effort)]]
        case (.anthropic, .enabled):
            return ["thinking": ["type": "adaptive", "display": "omitted"]]
        case (.anthropic, .disabled):
            return ["thinking": ["type": "disabled"]]
        case (.anthropic, .budget(let tokens)):
            return ["thinking": ["type": "enabled", "budget_tokens": .number(Double(tokens))]]
        case (.gptOSS, .effort(let effort)):
            return ["reasoning_effort": .string(effort)]
        case (.openAI, .effort(let effort)):
            return ["reasoning": ["effort": .string(effort)]]
        case (.openAI, .disabled):
            return ["reasoning": ["effort": "none"]]
        case (.nova, .effort(let effort)):
            return ["reasoningConfig": ["type": "enabled", "maxReasoningEffort": .string(effort)]]
        case (.nova, .enabled):
            return ["reasoningConfig": ["type": "enabled"]]
        case (.nova, .disabled):
            return ["reasoningConfig": ["type": "disabled"]]
        default:
            throw ChatError.unsupported(
                "No known Bedrock reasoning field for \(modelID) with \(reasoning). Pass additionalModelRequestFields instead."
            )
        }
    }

    /// Converse requires turns to alternate and to start with the user. The Assistant can leave
    /// two user turns in a row after a failed reply, so same-role neighbours are merged and a
    /// leading assistant turn is dropped (as VoiceInk's in-app client does).
    static func normalizedTurns(_ turns: [ChatRequest.Message]) -> [ChatRequest.Message] {
        var result: [ChatRequest.Message] = []
        for turn in turns where turn.role != .system {
            if result.isEmpty, turn.role == .assistant { continue }
            if let last = result.last, last.role == turn.role {
                result[result.count - 1].content = last.content + "\n\n" + turn.content
            } else {
                result.append(turn)
            }
        }
        return result
    }

    // MARK: - Response

    public static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(ResponseBody.self, from: data)
        let text = (response.output?.message?.content ?? [])
            .compactMap(\.text)
            .joined()
            // Some models end lines with Markdown hard-break spaces, which would be pasted.
            .replacingOccurrences(of: "[ \t]+(?=\n|$)", with: "", options: .regularExpression)
        return try ChatResult.make(
            text: text,
            finishReason: mapStopReason(response.stopReason),
            rawFinishReason: response.stopReason,
            usage: response.usage.map {
                ChatUsage(inputTokens: $0["inputTokens"]?.intValue, outputTokens: $0["outputTokens"]?.intValue)
            }
        )
    }

    static func mapStopReason(_ reason: String?) -> FinishReason {
        switch reason {
        case "end_turn", "stop_sequence": return .stop
        case "max_tokens", "model_context_window_exceeded": return .length
        case "guardrail_intervened", "content_filtered": return .contentFilter
        case "tool_use": return .toolCalls
        default: return .other
        }
    }

    private struct ResponseBody: Decodable {
        struct Output: Decodable {
            let message: Message?
        }

        struct Message: Decodable {
            let content: [ContentBlock]?
        }

        /// A union in the API. Only `text` is read; `reasoningContent` and the rest are skipped.
        struct ContentBlock: Decodable {
            let text: String?
        }

        let output: Output?
        let stopReason: String?
        let usage: JSONValue?
    }
}
