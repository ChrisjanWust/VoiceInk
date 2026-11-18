import Foundation
import LLMkit

/// Sends the in-app reasoning requests (Anthropic, Mistral, Ollama levels) with the same
/// timeout and retry rules as LLMkit's chat clients, and reports failures as `LLMKitError`
/// so they're handled like the Default requests LLMkit sends.
///
/// LLMkit's own `performRequest` is internal to the package, so this mirrors it: up to
/// three attempts, a 1 s then 2 s backoff for retryable statuses and network errors, and immediate
/// retries for timeouts (including HTTP 408) only when `retryOnTimeout` is on.
enum ReasoningChatHTTP {
    /// LLMkit's chat retry codes: 429 and every 5xx.
    static let chatRetryableStatusCodes: Set<Int> = Set([429]).union(500...599)

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

    static func jsonRequest(url: URL, body: [String: Any], headers: [String: String]) throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        for (field, value) in headers {
            request.setValue(value, forHTTPHeaderField: field)
        }
        guard JSONSerialization.isValidJSONObject(body),
            let data = try? JSONSerialization.data(withJSONObject: body)
        else {
            throw LLMKitError.encodingError
        }
        request.httpBody = data
        return request
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
