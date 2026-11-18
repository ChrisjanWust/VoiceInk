import Foundation

/// Errors from VoiceInkChat. The cases mirror LLMkit's `LLMKitError`, so the app can map
/// them the same way, plus ``unsupported(_:)`` and an error type on HTTP failures.
///
/// Cancellation is never wrapped: a cancelled task gets `CancellationError`.
public enum ChatError: Error, LocalizedError, Sendable, Equatable {
    /// The API key is missing or blank.
    case missingAPIKey
    case invalidURL(String)
    /// The request asks for something this provider's API can't express, such as
    /// `.disabled` reasoning on Gemini. Nothing was sent.
    case unsupported(String)
    /// No HTTP response: DNS, TLS, connection reset, offline.
    case networkError(String)
    /// A non-2xx response after any retries.
    /// - `message`: the provider's error message when the body has one, otherwise the body.
    /// - `errorType`: the provider's error type or code, such as `invalid_request_error`,
    ///   `RESOURCE_EXHAUSTED` or Bedrock's `ValidationException`.
    case httpError(statusCode: Int, message: String, errorType: String?)
    case decodingError(String)
    /// A successful response with no answer text and no truncation or filter to explain it.
    case noResultReturned
    case encodingError
    /// An attempt ran past its timeout, and no retry was allowed or left.
    case timeout

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "API key is missing or empty."
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .unsupported(let detail):
            return "Unsupported request: \(detail)"
        case .networkError(let detail):
            return "Network error: \(detail)"
        case .httpError(let statusCode, let message, _):
            return "HTTP \(statusCode): \(message)"
        case .decodingError(let detail):
            return "Failed to decode response: \(detail)"
        case .noResultReturned:
            return "The API returned no usable result."
        case .encodingError:
            return "Failed to encode request body."
        case .timeout:
            return "The request timed out."
        }
    }

    public var statusCode: Int? {
        if case .httpError(let statusCode, _, _) = self { return statusCode }
        return nil
    }
}

/// Reads the message and type out of a provider's error body.
enum ProviderErrorBody {
    /// Handles the shapes our providers use:
    /// - OpenAI, Groq, Anthropic, OpenRouter: `{"error": {"message", "type" | "code"}}`
    /// - Gemini: `{"error": {"message", "status"}}`
    /// - Ollama, some gateways: `{"error": "text"}`
    /// - Bedrock: `{"message" | "Message", "__type"}`, with the type usually in a header
    /// - Mistral and others: `{"message", "type"}` or `{"detail": ...}`
    ///
    /// Falls back to the raw body (or "No error details" when it's empty).
    static func parse(_ data: Data, errorTypeHeader: String? = nil) -> (message: String, errorType: String?) {
        let raw = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var message: String?
        var type: String?

        if let json = try? JSONValue(data: data), let object = json.objectValue {
            if let error = object["error"] {
                if let text = error.stringValue {
                    message = text
                } else if let error = error.objectValue {
                    message = error["message"]?.stringValue
                    type =
                        error["type"]?.stringValue ?? error["code"]?.stringValue
                        ?? error["status"]?.stringValue ?? error["code"]?.intValue.map(String.init)
                }
            }
            message = message ?? object["message"]?.stringValue ?? object["Message"]?.stringValue
            if message == nil, let detail = object["detail"] {
                message = detail.stringValue ?? (try? detail.encoded()).flatMap { String(data: $0, encoding: .utf8) }
            }
            type = type ?? object["__type"]?.stringValue ?? object["type"]?.stringValue
                ?? object["code"]?.stringValue
        }

        if let errorTypeHeader, !errorTypeHeader.isEmpty {
            type = errorTypeHeader
        }
        let normalizedType = type.map(normalizedErrorType).flatMap { $0.isEmpty ? nil : $0 }
        let finalMessage = (message?.isEmpty == false ? message : nil) ?? (raw.isEmpty ? "No error details" : raw)
        return (finalMessage, normalizedType)
    }

    /// Turns AWS's `AccessDeniedException:http://...` or `com.amazonaws.bedrock#AccessDeniedException`
    /// into `AccessDeniedException`. Other types pass through unchanged.
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
