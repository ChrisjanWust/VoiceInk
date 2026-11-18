import Foundation
import LLMkit

/// Shared HTTP transport for VoiceInk's in-app chat clients.
///
/// It sends a request with the same timeout and retry rules as LLMkit's chat clients and
/// reports failures as `LLMKitError`, so callers handle them exactly like errors from LLMkit:
/// - up to three attempts,
/// - a 1 s then 2 s backoff for retryable statuses and network errors,
/// - immediate retries for timeouts (including HTTP 408), only when `retryOnTimeout` is on,
/// - `LLMKitError.httpError` with the response body for any other non-2xx status.
///
/// LLMkit's own `performRequest` is internal to the package, so this mirrors it.
enum ChatHTTP {
    /// LLMkit's chat retry statuses: 429 and every 5xx.
    static let chatRetryableStatusCodes: Set<Int> = Set([429]).union(500...599)
    /// LLMkit's default retry statuses, which its Ollama client uses.
    static let defaultRetryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]

    /// A POST request with a JSON body.
    static func jsonRequest(url: URL, body: [String: Any], headers: [String: String] = [:]) throws -> URLRequest {
        guard JSONSerialization.isValidJSONObject(body),
            let data = try? JSONSerialization.data(withJSONObject: body)
        else {
            throw LLMKitError.encodingError
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        for (field, value) in headers {
            request.setValue(value, forHTTPHeaderField: field)
        }
        request.httpBody = data
        return request
    }

    /// Decodes a response body, reporting failures as `LLMKitError.decodingError`.
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw LLMKitError.decodingError(error.localizedDescription)
        }
    }

    /// Sends the request and returns the body of a 2xx response.
    static func send(
        _ request: URLRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool,
        retryableStatusCodes: Set<Int> = chatRetryableStatusCodes,
        maxRetries: Int = 2
    ) async throws -> Data {
        var request = request
        request.timeoutInterval = timeout
        let maxRetries = max(0, maxRetries)
        var lastError: Error?
        var delayBeforeRetry = true

        for attempt in 0...maxRetries {
            try Task.checkCancellation()
            if attempt > 0 && delayBeforeRetry {
                try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt - 1)) * 1_000_000_000))
            }

            let session = makeSession(timeout: timeout)
            defer { session.finishTasksAndInvalidate() }

            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw LLMKitError.networkError("No HTTP response received.")
                }
                let retryable =
                    retryableStatusCodes.contains(http.statusCode) || (retryOnTimeout && http.statusCode == 408)
                if retryable && attempt < maxRetries {
                    delayBeforeRetry = http.statusCode != 408
                    lastError = LLMKitError.httpError(
                        statusCode: http.statusCode,
                        message: String(data: data, encoding: .utf8) ?? ""
                    )
                    continue
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw LLMKitError.httpError(
                        statusCode: http.statusCode,
                        message: String(data: data, encoding: .utf8) ?? "No error details"
                    )
                }
                return data
            } catch let error as LLMKitError {
                throw error
            } catch let error as NSError where error.domain == NSURLErrorDomain && error.code == NSURLErrorTimedOut {
                guard retryOnTimeout, attempt < maxRetries else { throw LLMKitError.timeout }
                lastError = LLMKitError.timeout
                delayBeforeRetry = false
            } catch {
                try Task.checkCancellation()
                let nsError = error as NSError
                if error is CancellationError
                    || (nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled)
                {
                    throw error
                }
                lastError = LLMKitError.networkError(error.localizedDescription)
                delayBeforeRetry = true
            }
        }

        throw lastError ?? LLMKitError.networkError("Request failed after \(maxRetries + 1) attempts")
    }

    private static func makeSession(timeout: TimeInterval) -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }
}
