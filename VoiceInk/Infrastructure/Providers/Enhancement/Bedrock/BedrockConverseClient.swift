import Foundation

/// The AWS Region that Amazon Bedrock requests go to.
///
/// Bedrock API keys are sent as bearer tokens, so the Region is the only other piece of
/// configuration a request needs. Short-term keys only work in the Region they were made in.
enum BedrockRegion {
    static let userDefaultsKey = "bedrockRegion"
    static let defaultRegion = "us-east-1"

    /// Regions offered in the picker. Any other valid Region code can be typed in.
    static let commonRegions = [
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "ca-central-1",
        "eu-central-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-north-1",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-south-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "sa-east-1",
    ]

    static var current: String {
        get {
            let saved = UserDefaults.standard.string(forKey: userDefaultsKey).map(normalized) ?? ""
            return saved.isEmpty ? defaultRegion : saved
        }
        set {
            let previous = current
            let value = normalized(newValue)
            if value.isEmpty {
                UserDefaults.standard.removeObject(forKey: userDefaultsKey)
            } else {
                UserDefaults.standard.set(value, forKey: userDefaultsKey)
            }
            if current != previous {
                BedrockModelCatalog.regionDidChange()
            }
        }
    }

    static func normalized(_ region: String) -> String {
        region.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Matches Region codes such as `us-east-1`, `eu-central-2` and `us-gov-west-1`.
    static func isValid(_ region: String) -> Bool {
        normalized(region).range(of: #"^[a-z]{2}(-[a-z]+)+-[0-9]{1,2}$"#, options: .regularExpression) != nil
    }

    static func runtimeEndpoint(for region: String) -> String {
        "https://bedrock-runtime.\(normalized(region)).amazonaws.com"
    }
}

struct BedrockMessage: Sendable, Equatable {
    enum Role: String, Sendable {
        case user
        case assistant
    }

    let role: Role
    let text: String
}

struct BedrockConverseResult: Sendable, Equatable {
    /// The model's answer: every `text` content block, joined. Reasoning blocks are never included.
    let text: String
    let stopReason: String?

    var wasTruncated: Bool {
        stopReason == "max_tokens" || stopReason == "model_context_window_exceeded"
    }

    var wasBlocked: Bool {
        stopReason == "guardrail_intervened" || stopReason == "content_filtered"
    }
}

/// Errors from Amazon Bedrock, with messages written for the person using the app.
enum BedrockError: Error, LocalizedError, Equatable {
    case missingAPIKey
    case invalidRegion(String)
    case missingModelID
    case unreachableRegion(region: String, detail: String)
    case networkError(String)
    case timeout
    case invalidResponse(String)
    case contentBlocked(stopReason: String)
    case outputTruncated
    case http(statusCode: Int, errorType: String?, message: String, modelID: String, region: String)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Enter an Amazon Bedrock API key."
        case .invalidRegion(let region):
            return
                "\"\(region)\" isn't a valid AWS Region code. Use a code such as us-east-1 or eu-central-1."
        case .missingModelID:
            return "Choose an Amazon Bedrock model, or enter a model ID or inference profile ID."
        case .unreachableRegion(let region, let detail):
            return
                "Couldn't reach Amazon Bedrock at \(BedrockRegion.runtimeEndpoint(for: region)). Check that \(region) is a Region where Bedrock is available. (\(detail))"
        case .networkError(let detail):
            return "Couldn't connect to Amazon Bedrock: \(detail)"
        case .timeout:
            return "Amazon Bedrock didn't respond in time."
        case .invalidResponse(let detail):
            return "Amazon Bedrock sent a response VoiceInk couldn't read: \(detail)"
        case .contentBlocked(let stopReason):
            return stopReason == "guardrail_intervened"
                ? "An Amazon Bedrock guardrail blocked this response."
                : "Amazon Bedrock filtered this response."
        case .outputTruncated:
            return "The Amazon Bedrock model stopped before finishing its response."
        case .http(let statusCode, let errorType, let message, let modelID, let region):
            return BedrockErrorDescriber.describe(
                statusCode: statusCode,
                errorType: errorType,
                message: message,
                modelID: modelID,
                region: region
            )
        }
    }

    var isRetryable: Bool {
        switch self {
        case .networkError:
            return true
        case .http(let statusCode, let errorType, _, _, _):
            if statusCode == 429 {
                return true
            }
            if errorType == "ModelErrorException" {
                return false
            }
            return [500, 502, 503, 504, 529].contains(statusCode)
        default:
            return false
        }
    }
}

enum BedrockErrorDescriber {
    /// Prefixes of system-defined cross-Region inference profiles, such as `us.` or `global.`.
    static let inferenceProfilePrefixes = ["global", "us", "us-gov", "eu", "apac", "au", "ca", "jp"]

    static func isInferenceProfileID(_ modelID: String) -> Bool {
        guard let prefix = modelID.split(separator: ".", maxSplits: 1).first,
            modelID.contains(".")
        else {
            return false
        }
        return inferenceProfilePrefixes.contains(String(prefix).lowercased())
    }

    static func describe(
        statusCode: Int,
        errorType: String?,
        message: String,
        modelID: String,
        region: String
    ) -> String {
        let type = errorType ?? ""
        let lowercasedMessage = message.lowercased()
        let awsSaid = message.isEmpty ? "" : " AWS said: \(message)"

        let authErrorTypes: Set<String> = [
            "UnrecognizedClientException",
            "InvalidClientTokenId",
            "ExpiredTokenException",
            "InvalidSignatureException",
            "IncompleteSignature",
            "MissingAuthenticationTokenException",
        ]
        let authPhrases = [
            "api key", "bearer", "security token", "authentication", "token has expired", "token is expired",
            "signature",
        ]
        if authErrorTypes.contains(type) || statusCode == 401
            || (statusCode == 403 && authPhrases.contains(where: { lowercasedMessage.contains($0) }))
        {
            return
                "Amazon Bedrock didn't accept this API key. Check that it's complete and hasn't expired. Short-term keys expire within 12 hours and only work in the Region where they were created (VoiceInk is using \(region))."
                + awsSaid
        }

        if type == "FTUFormNotFilled" || lowercasedMessage.contains("use case details") {
            return
                "Your AWS account must submit Anthropic's use case details before it can use \(modelID). Open a Claude model in the Amazon Bedrock console's model catalog to submit them, then try again."
                + awsSaid
        }

        if lowercasedMessage.contains("marketplace") || lowercasedMessage.contains("subscription")
            || lowercasedMessage.contains("agreement")
        {
            return
                "Your AWS account can't use \(modelID) yet because its AWS Marketplace subscription isn't active. New subscriptions can take about 15 minutes. If it keeps failing, open the model in the Amazon Bedrock console to check the subscription."
                + awsSaid
        }

        if type == "AccessDeniedException" || statusCode == 403 {
            return
                "Amazon Bedrock denied access to \(modelID) in \(region). Check that your AWS account can use this model in this Region, and that the API key's IAM permissions allow bedrock:InvokeModel for it."
                + awsSaid
        }

        if lowercasedMessage.contains("on-demand throughput") || lowercasedMessage.contains("inference profile") {
            if isInferenceProfileID(modelID) {
                return
                    "\(modelID) can't be used from \(region). Pick an inference profile that this Region supports. global. profiles work from most commercial Regions; us., eu. and other geographic profiles only work from Regions in that geography."
                    + awsSaid
            }
            return
                "\(modelID) can't be called directly in \(region). Use its inference profile ID instead, such as global.\(modelID) or us.\(modelID)."
                + awsSaid
        }

        if lowercasedMessage.contains("end of its life") || lowercasedMessage.contains("end of life") {
            return
                "\(modelID) has been retired on Amazon Bedrock. Choose a current model."
                + awsSaid
        }

        if type == "ResourceNotFoundException" || statusCode == 404
            || lowercasedMessage.contains("model identifier is invalid")
            || lowercasedMessage.contains("model id is invalid")
        {
            return
                "\(modelID) isn't available in \(region). Check the model ID, or switch to a Region that offers this model. Geographic inference profiles such as us. or eu. only work from Regions in that geography."
                + awsSaid
        }

        if type == "ModelNotReadyException" {
            return "\(modelID) isn't ready to serve requests yet. Try again in a moment." + awsSaid
        }

        if statusCode == 429 || type == "ThrottlingException" {
            return
                "Amazon Bedrock is throttling requests to \(modelID) because your account reached its quota. Try again shortly."
                + awsSaid
        }

        if statusCode == 408 || type == "ModelTimeoutException" {
            return "\(modelID) took too long to respond." + awsSaid
        }

        if statusCode == 424 || type == "ModelErrorException" {
            return "\(modelID) failed while processing the request." + awsSaid
        }

        if (500...599).contains(statusCode) {
            return "Amazon Bedrock had a server problem (HTTP \(statusCode)). Try again later." + awsSaid
        }

        if statusCode == 400 || type == "ValidationException" {
            return "Amazon Bedrock rejected the request for \(modelID)." + awsSaid
        }

        return "Amazon Bedrock returned HTTP \(statusCode)." + awsSaid
    }

    /// Reads the error type and message from a Bedrock error response.
    ///
    /// Bedrock sends `{"message": "..."}` (sometimes `Message`) and names the error in the
    /// `x-amzn-ErrorType` header, for example `AccessDeniedException:http://internal.amazon.com/...`.
    static func parse(data: Data, errorTypeHeader: String?) -> (errorType: String?, message: String) {
        var message = ""
        var bodyType: String?

        if let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            message = (object["message"] as? String) ?? (object["Message"] as? String) ?? ""
            bodyType = (object["__type"] as? String) ?? (object["code"] as? String)
        } else if let text = String(data: data, encoding: .utf8) {
            message = text
        }

        let errorType = (errorTypeHeader ?? bodyType).map(normalizedErrorType)
        return (errorType?.isEmpty == false ? errorType : nil, message.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Turns `AccessDeniedException:http://...` or `com.amazonaws.bedrock#AccessDeniedException` into `AccessDeniedException`.
    static func normalizedErrorType(_ raw: String) -> String {
        var value = raw
        if let colon = value.firstIndex(of: ":") {
            value = String(value[..<colon])
        }
        if let hash = value.lastIndex(of: "#") {
            value = String(value[value.index(after: hash)...])
        }
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Minimal client for the Amazon Bedrock Runtime Converse API, authenticated with a Bedrock API key.
///
/// `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse`
///
/// By default no inference or reasoning parameters are sent, so each model runs with its Bedrock
/// defaults. A picked reasoning level adds that model's own `additionalModelRequestFields`.
/// Only `text` content blocks are returned; `reasoningContent` blocks are dropped.
enum BedrockConverseClient {
    static func converse(
        apiKey: String,
        region: String,
        modelID: String,
        systemPrompt: String?,
        messages: [BedrockMessage],
        reasoning: BedrockReasoningFields? = nil,
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false,
        maxRetries: Int = 2
    ) async throws -> BedrockConverseResult {
        let request = try makeRequest(
            apiKey: apiKey,
            region: region,
            modelID: modelID,
            systemPrompt: systemPrompt,
            messages: messages,
            reasoning: reasoning,
            timeout: timeout
        )
        let normalizedRegion = BedrockRegion.normalized(region)
        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)

        var lastError: Error = BedrockError.networkError("The request failed.")
        let attempts = max(0, maxRetries) + 1
        var delayBeforeRetry = true

        for attempt in 0..<attempts {
            try Task.checkCancellation()
            if attempt > 0 && delayBeforeRetry {
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt - 1)) * 1_000_000_000))
            }

            do {
                let (data, response) = try await send(request, timeout: timeout, region: normalizedRegion)
                return try parseResponse(
                    data: data,
                    response: response,
                    modelID: trimmedModelID,
                    region: normalizedRegion
                )
            } catch let error as BedrockError {
                lastError = error
                let isTimeout =
                    error == .timeout
                    || {
                        if case .http(408, _, _, _, _) = error { return true }
                        return false
                    }()
                if isTimeout {
                    guard retryOnTimeout, attempt < attempts - 1 else { throw error }
                    delayBeforeRetry = false
                    continue
                }
                guard error.isRetryable, attempt < attempts - 1 else { throw error }
                delayBeforeRetry = true
            }
        }

        throw lastError
    }

    /// Checks the key, Region and model together with a tiny Converse request.
    ///
    /// A bearer key has no cheaper self-check that's guaranteed to be allowed by its IAM policy,
    /// and this also catches models the account can't use and Region mismatches up front.
    static func verifyAPIKey(
        _ apiKey: String,
        region: String,
        modelID: String,
        timeout: TimeInterval = 20
    ) async -> (isValid: Bool, errorMessage: String?) {
        do {
            _ = try await converse(
                apiKey: apiKey,
                region: region,
                modelID: modelID,
                systemPrompt: nil,
                messages: [BedrockMessage(role: .user, text: "Reply with the single word OK.")],
                timeout: timeout,
                retryOnTimeout: false,
                maxRetries: 0
            )
            return (true, nil)
        } catch BedrockError.outputTruncated {
            // The key, Region and model all worked; the reply just hit the model's default limit.
            return (true, nil)
        } catch BedrockError.contentBlocked {
            return (true, nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }

    static func normalizedAPIKey(_ apiKey: String) -> String {
        var key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if key.lowercased().hasPrefix("bearer ") {
            key = String(key.dropFirst("bearer ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return key
    }

    /// Percent-encodes a model ID, inference profile ID or ARN for the request path.
    static func encodedModelID(_ modelID: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return modelID.addingPercentEncoding(withAllowedCharacters: allowed) ?? modelID
    }

    static func makeRequest(
        apiKey: String,
        region: String,
        modelID: String,
        systemPrompt: String?,
        messages: [BedrockMessage],
        reasoning: BedrockReasoningFields? = nil,
        timeout: TimeInterval
    ) throws -> URLRequest {
        let key = normalizedAPIKey(apiKey)
        guard !key.isEmpty else { throw BedrockError.missingAPIKey }

        let normalizedRegion = BedrockRegion.normalized(region)
        guard BedrockRegion.isValid(normalizedRegion) else { throw BedrockError.invalidRegion(region) }

        let trimmedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedModelID.isEmpty else { throw BedrockError.missingModelID }

        let urlString =
            "\(BedrockRegion.runtimeEndpoint(for: normalizedRegion))/model/\(encodedModelID(trimmedModelID))/converse"
        guard let url = URL(string: urlString) else { throw BedrockError.missingModelID }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(
            ConverseRequestBody(systemPrompt: systemPrompt, messages: messages, reasoning: reasoning)
        )
        return request
    }

    static func parseResponse(
        data: Data,
        response: URLResponse,
        modelID: String,
        region: String
    ) throws -> BedrockConverseResult {
        guard let http = response as? HTTPURLResponse else {
            throw BedrockError.invalidResponse("No HTTP response.")
        }

        guard (200..<300).contains(http.statusCode) else {
            let parsed = BedrockErrorDescriber.parse(
                data: data,
                errorTypeHeader: http.value(forHTTPHeaderField: "x-amzn-ErrorType")
            )
            throw BedrockError.http(
                statusCode: http.statusCode,
                errorType: parsed.errorType,
                message: parsed.message,
                modelID: modelID,
                region: region
            )
        }

        let result = try decodeResult(from: data)
        if result.wasBlocked {
            throw BedrockError.contentBlocked(stopReason: result.stopReason ?? "")
        }
        if result.wasTruncated {
            throw BedrockError.outputTruncated
        }
        return result
    }

    static func decodeResult(from data: Data) throws -> BedrockConverseResult {
        let decoded: ConverseResponseBody
        do {
            decoded = try JSONDecoder().decode(ConverseResponseBody.self, from: data)
        } catch {
            throw BedrockError.invalidResponse(error.localizedDescription)
        }

        let text = (decoded.output?.message?.content ?? [])
            .compactMap(\.text)
            .joined()
            // Some models end lines with Markdown hard-break spaces, which would be pasted.
            .replacingOccurrences(of: "[ \t]+(?=\n|$)", with: "", options: .regularExpression)
        return BedrockConverseResult(text: text, stopReason: decoded.stopReason)
    }

    private static func send(
        _ request: URLRequest,
        timeout: TimeInterval,
        region: String
    ) async throws -> (Data, URLResponse) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        do {
            return try await session.data(for: request)
        } catch let error as NSError where error.domain == NSURLErrorDomain {
            switch error.code {
            case NSURLErrorCancelled:
                throw CancellationError()
            case NSURLErrorTimedOut:
                throw BedrockError.timeout
            case NSURLErrorCannotFindHost, NSURLErrorDNSLookupFailed:
                throw BedrockError.unreachableRegion(region: region, detail: error.localizedDescription)
            default:
                throw BedrockError.networkError(error.localizedDescription)
            }
        }
    }
}

// MARK: - Request and response bodies

private struct ConverseRequestBody: Encodable {
    struct TextBlock: Encodable {
        let text: String
    }

    struct Message: Encodable {
        let role: String
        let content: [TextBlock]
    }

    struct InferenceConfig: Encodable {
        let maxTokens: Int
    }

    let messages: [Message]
    let system: [TextBlock]?
    // Left out entirely on Default, so the request matches one with no reasoning level.
    let additionalModelRequestFields: [String: BedrockJSONValue]?
    let inferenceConfig: InferenceConfig?

    init(systemPrompt: String?, messages: [BedrockMessage], reasoning: BedrockReasoningFields? = nil) {
        self.messages = messages.map { Message(role: $0.role.rawValue, content: [TextBlock(text: $0.text)]) }
        if let systemPrompt, !systemPrompt.isEmpty {
            self.system = [TextBlock(text: systemPrompt)]
        } else {
            self.system = nil
        }
        self.additionalModelRequestFields = reasoning?.additionalModelRequestFields
        self.inferenceConfig = reasoning?.maxTokens.map(InferenceConfig.init(maxTokens:))
    }
}

private struct ConverseResponseBody: Decodable {
    struct Output: Decodable {
        let message: Message?
    }

    struct Message: Decodable {
        let role: String?
        let content: [ContentBlock]?
    }

    /// A union in the API. Only `text` is read; `reasoningContent`, `toolUse`,
    /// `citationsContent` and the rest are skipped.
    struct ContentBlock: Decodable {
        let text: String?
    }

    let output: Output?
    let stopReason: String?
}
