// Request and response shapes ported from teunlao/swift-ai-sdk (commit 35c334a4, Apache-2.0;
// see LICENSE-APACHE-2.0 in this package):
// - Sources/AmazonBedrockProvider/BedrockChatLanguageModel.swift (Converse body,
//   additionalModelRequestFields)
// - Sources/AmazonBedrockProvider/MapBedrockFinishReason.swift
// - Sources/AmazonBedrockProvider/BedrockProvider.swift (bearer API keys)
// Modified for VoiceInkChat: text-only, no tools, SigV4, guardrails, reasoning mapping or
// streaming; the message normalisation and trailing-space cleanup from VoiceInk's in-app
// BedrockConverseClient.

import Foundation

/// A client for the Amazon Bedrock Runtime Converse API, authenticated with a Bedrock API key
/// (a bearer token).
///
/// `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse`
///
/// Reasoning fields differ per model and are read from each model's schema, so the caller
/// passes them in `additionalModelRequestFields` (see ``complete(_:additionalModelRequestFields:)``),
/// and ``ChatRequest/reasoning`` throws ``ChatError/unsupported(_:)``. ``ChatRequest/extraBody`` is
/// merged over the top-level Converse body, which is where Fast mode's `performanceConfig` goes.
///
/// Only `text` blocks reach the answer; `reasoningContent` blocks and inline `<think>` sections
/// are dropped.
public struct BedrockChatClient: ChatClient {
    /// Statuses the in-app client retries: throttling and server errors, including 529.
    /// 424 (`ModelErrorException`) isn't retried, since the model itself failed.
    static let retryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504, 529]

    public var apiKey: String
    public var region: String
    var http = ChatHTTP.shared

    public init(apiKey: String, region: String) {
        self.apiKey = apiKey
        self.region = region
    }

    public func complete(_ request: ChatRequest) async throws -> ChatResult {
        try await complete(request, additionalModelRequestFields: [:])
    }

    /// Sends the request with model-specific fields, such as `reasoning_config` or
    /// `output_config`, in `additionalModelRequestFields`.
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

    static func normalizedRegion(_ region: String) -> String {
        region.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Accepts keys pasted with a `Bearer ` prefix.
    static func normalizedAPIKey(_ apiKey: String) -> String {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let prefix = key.range(of: #"^bearer(\s+|$)"#, options: [.regularExpression, .caseInsensitive]) else {
            return key
        }
        return String(key[prefix.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Region codes such as `us-east-1`, `eu-central-2` and `us-gov-west-1`.
    static func isValidRegion(_ region: String) -> Bool {
        normalizedRegion(region).range(of: #"^[a-z]{2}(-[a-z]+)+-[0-9]{1,2}$"#, options: .regularExpression) != nil
    }

    static func endpoint(region: String, modelID: String) throws -> URL {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        let encodedModelID = modelID.addingPercentEncoding(withAllowedCharacters: allowed) ?? modelID
        let string = "https://bedrock-runtime.\(normalizedRegion(region)).amazonaws.com/model/\(encodedModelID)/converse"
        guard let url = URL(string: string) else { throw ChatError.invalidURL(string) }
        return url
    }

    func makeURLRequest(
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

        let headers = ChatHTTP.mergedHeaders(
            ["Accept": "application/json"], request.extraHeaders, ["Authorization": "Bearer \(key)"])
        return try ChatHTTP.jsonRequest(
            url: try Self.endpoint(region: region, modelID: modelID),
            body: try Self.body(for: request, additionalModelRequestFields: additionalModelRequestFields),
            headers: headers
        )
    }

    static func body(
        for request: ChatRequest,
        additionalModelRequestFields: [String: JSONValue] = [:]
    ) throws -> [String: JSONValue] {
        if request.reasoning != nil {
            throw ChatError.unsupported(
                "Bedrock reasoning fields differ per model. Pass them in additionalModelRequestFields.")
        }
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
        if !inference.isEmpty { body["inferenceConfig"] = .object(inference) }
        if !additionalModelRequestFields.isEmpty {
            body["additionalModelRequestFields"] = .object(additionalModelRequestFields)
        }
        return body.merging(extra: request.extraBody)
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

    static func decodeResult(from data: Data) throws -> ChatResult {
        let response = try ChatHTTP.decode(ResponseBody.self, from: data)
        let text = ReasoningText.strip((response.output?.message?.content ?? []).compactMap(\.text).joined())
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
