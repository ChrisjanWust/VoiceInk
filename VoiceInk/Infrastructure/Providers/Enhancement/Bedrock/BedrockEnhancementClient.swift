import Foundation
import VoiceInkChat

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

    /// A rejected or expired API key, as opposed to a model or permission problem.
    static func isAuthError(statusCode: Int, errorType: String?, message: String) -> Bool {
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
        let lowercasedMessage = message.lowercased()
        return authErrorTypes.contains(errorType ?? "") || statusCode == 401
            || (statusCode == 403 && authPhrases.contains(where: { lowercasedMessage.contains($0) }))
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

        if isAuthError(statusCode: statusCode, errorType: errorType, message: message) {
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

/// VoiceInkChat's Bedrock Converse client with a model's reasoning fields, reporting failures as
/// `BedrockError` so people see VoiceInk's Bedrock messages.
///
/// Default sends no inference or reasoning parameters, so each model runs with its Bedrock
/// defaults. A picked reasoning level adds that model's own `additionalModelRequestFields`.
struct BedrockEnhancementClient: ChatClient {
    let apiKey: String
    let region: String
    var additionalModelRequestFields: [String: JSONValue] = [:]

    /// Throws `BedrockError`, including `outputTruncated` and `contentBlocked` for a cut-off or
    /// guardrail-replaced answer, or `ChatError.noResultReturned` for an empty one.
    func complete(_ request: ChatRequest) async throws -> ChatResult {
        let modelID = request.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let region = BedrockRegion.normalized(region)
        let result: ChatResult
        do {
            result = try await BedrockChatClient(apiKey: apiKey, region: region)
                .complete(request, additionalModelRequestFields: additionalModelRequestFields)
        } catch let error as ChatError {
            throw Self.bedrockError(for: error, modelID: modelID, region: region)
        }
        switch result.finishReason {
        case .contentFilter:
            throw BedrockError.contentBlocked(stopReason: result.rawFinishReason ?? "")
        case .length:
            throw BedrockError.outputTruncated
        default:
            return result
        }
    }

    private static func bedrockError(for error: ChatError, modelID: String, region: String) -> Error {
        switch error {
        case .missingAPIKey:
            return BedrockError.missingAPIKey
        case .invalidURL:
            return BedrockError.invalidRegion(region)
        case .unsupported:
            return BedrockError.missingModelID
        case .networkError(let detail, let code):
            if code == .cannotFindHost || code == .dnsLookupFailed {
                return BedrockError.unreachableRegion(region: region, detail: detail)
            }
            return BedrockError.networkError(detail)
        case .timeout:
            return BedrockError.timeout
        case .httpError(let statusCode, let message, let errorType):
            return BedrockError.http(
                statusCode: statusCode,
                errorType: errorType,
                message: message,
                modelID: modelID,
                region: region
            )
        case .decodingError(let detail):
            return BedrockError.invalidResponse(detail)
        case .encodingError:
            return BedrockError.invalidResponse(error.localizedDescription)
        case .noResultReturned:
            return error
        }
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
            _ = try await BedrockEnhancementClient(apiKey: apiKey, region: region).complete(
                ChatRequest(model: modelID, messages: [.user("Reply with the single word OK.")], timeout: timeout)
            )
            return (true, nil)
        } catch BedrockError.outputTruncated, BedrockError.contentBlocked, ChatError.noResultReturned {
            // The key, Region and model all worked; only the reply was cut off, filtered or empty.
            return (true, nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }
}
