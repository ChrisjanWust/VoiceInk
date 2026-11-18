import Foundation

/// The shared HTTP transport for every chat client.
///
/// It keeps the retry and timeout rules of LLMkit's chat clients (and of the app's in-app
/// `ChatHTTP`), so switching a call site over doesn't change how failures behave:
/// - up to three attempts (`maxRetries: 2`);
/// - a 1 s then 2 s backoff for retryable statuses (429 and every 5xx by default) and for
///   network errors;
/// - timeouts, including HTTP 408, retried immediately and only when `retryOnTimeout` is on;
/// - the timeout applies to each attempt, as a limit on the whole attempt, not just idle time;
/// - any other non-2xx status throws ``ChatError/httpError(statusCode:message:errorType:)``;
/// - cancellation passes straight through as `CancellationError`.
///
/// One difference, on purpose: LLMkit makes a new ephemeral `URLSession` for every attempt.
/// ``shared`` reuses one session, so requests to the same host keep the connection alive and
/// skip a fresh TCP and TLS handshake.
public struct ChatHTTP: Sendable {
    /// LLMkit's chat retry statuses: 429 and every 5xx.
    public static let chatRetryableStatusCodes: Set<Int> = Set([429]).union(500...599)
    /// LLMkit's default retry statuses, which its Ollama client uses.
    public static let defaultRetryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]

    /// One transport for the whole process, so connections are reused across requests.
    public static let shared = ChatHTTP()

    public let session: URLSession
    /// The first retry waits this long and each later one twice as long as the one before.
    let retryBaseDelay: TimeInterval

    /// - Parameters:
    ///   - session: Defaults to one session per process, made by ``makeSession(configuration:)``.
    ///   - retryBaseDelay: The backoff before the first retry, in seconds. Tests shorten it.
    public init(session: URLSession = ChatHTTP.sharedSession, retryBaseDelay: TimeInterval = 1) {
        self.session = session
        self.retryBaseDelay = retryBaseDelay
    }

    public static let sharedSession = makeSession()

    /// A session for chat requests: ephemeral (nothing written to disk), no cache, no cookies.
    /// Per-attempt timeouts are enforced by ``send(_:timeout:retryOnTimeout:retryableStatusCodes:maxRetries:errorTypeHeader:)``,
    /// so the session's own resource timeout is only a backstop.
    public static func makeSession(configuration: URLSessionConfiguration = .ephemeral) -> URLSession {
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        configuration.httpShouldSetCookies = false
        configuration.httpCookieAcceptPolicy = .never
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForResource = 600
        return URLSession(configuration: configuration)
    }

    // MARK: - Building requests

    /// A POST with a JSON body. Keys are sorted, so the same body always encodes the same way.
    public static func jsonRequest(
        url: URL,
        body: [String: JSONValue],
        headers: [String: String] = [:]
    ) throws -> URLRequest {
        let data: Data
        do {
            data = try JSONValue.object(body).encoded()
        } catch {
            throw ChatError.encodingError
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

    /// Decodes a response body, reporting failures as ``ChatError/decodingError(_:)``.
    public static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw ChatError.decodingError(String(describing: error))
        }
    }

    // MARK: - Sending

    /// Sends the request and returns the body of a 2xx response.
    ///
    /// - Parameter errorTypeHeader: A response header that names the error type, such as
    ///   Bedrock's `x-amzn-ErrorType`. It wins over a type found in the body.
    public func send(
        _ request: URLRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool,
        retryableStatusCodes: Set<Int> = ChatHTTP.chatRetryableStatusCodes,
        maxRetries: Int = 2,
        errorTypeHeader: String? = nil
    ) async throws -> Data {
        var timedRequest = request
        timedRequest.timeoutInterval = timeout
        let request = timedRequest
        let session = self.session
        return try await withRetries(
            timeout: timeout,
            retryOnTimeout: retryOnTimeout,
            retryableStatusCodes: retryableStatusCodes,
            maxRetries: maxRetries,
            errorTypeHeader: errorTypeHeader
        ) {
            let (data, response) = try await Self.withAttemptTimeout(timeout) {
                try await session.data(for: request)
            }
            guard let http = response as? HTTPURLResponse else {
                throw ChatError.networkError("No HTTP response received.")
            }
            return Attempt(status: http.statusCode, headers: http, payload: data, errorBody: { data })
        }
    }

    /// Sends a streaming request and returns its server-sent events.
    ///
    /// Retries and the per-attempt timeout apply until the response headers arrive. After that
    /// the stream isn't retried, and `timeout` is the longest gap allowed between chunks.
    public func events(
        _ request: URLRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool,
        retryableStatusCodes: Set<Int> = ChatHTTP.chatRetryableStatusCodes,
        maxRetries: Int = 2
    ) -> AsyncThrowingStream<ServerSentEvent, Error> {
        var streamRequest = request
        streamRequest.timeoutInterval = timeout
        streamRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let request = streamRequest
        let session = self.session
        let http = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let bytes = try await http.withRetries(
                        timeout: timeout,
                        retryOnTimeout: retryOnTimeout,
                        retryableStatusCodes: retryableStatusCodes,
                        maxRetries: maxRetries,
                        errorTypeHeader: nil
                    ) {
                        let stream = try await Self.withAttemptTimeout(timeout) {
                            let (bytes, response) = try await session.bytes(for: request)
                            return UncheckedBytes(bytes: bytes, response: response)
                        }
                        guard let response = stream.response as? HTTPURLResponse else {
                            throw ChatError.networkError("No HTTP response received.")
                        }
                        return Attempt(status: response.statusCode, headers: response, payload: stream) {
                            var body = Data()
                            for try await byte in stream.bytes {
                                body.append(byte)
                                if body.count > 64 * 1024 { break }
                            }
                            return body
                        }
                    }

                    var parser = ServerSentEventParser()
                    var line = Data()
                    for try await byte in bytes.bytes {
                        line.append(byte)
                        if byte == 0x0A || byte == 0x0D {
                            for event in parser.feed(line) { continuation.yield(event) }
                            line.removeAll(keepingCapacity: true)
                        }
                    }
                    for event in parser.feed(line) + parser.finish() { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.mapTransportError(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Retry loop

    private struct Attempt<Payload: Sendable>: Sendable {
        let status: Int
        let headers: HTTPURLResponse
        let payload: Payload
        let errorBody: @Sendable () async throws -> Data
    }

    private struct AttemptTimedOut: Error {}

    private func withRetries<Payload: Sendable>(
        timeout: TimeInterval,
        retryOnTimeout: Bool,
        retryableStatusCodes: Set<Int>,
        maxRetries: Int,
        errorTypeHeader: String?,
        attempt makeAttempt: @Sendable () async throws -> Attempt<Payload>
    ) async throws -> Payload {
        let maxRetries = max(0, maxRetries)
        var lastError: Error?
        var delayBeforeRetry = true

        for attempt in 0...maxRetries {
            try Task.checkCancellation()
            if attempt > 0 && delayBeforeRetry {
                let seconds = retryBaseDelay * pow(2.0, Double(attempt - 1))
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            }

            do {
                let result = try await makeAttempt()
                if (200..<300).contains(result.status) {
                    return result.payload
                }
                let body = (try? await result.errorBody()) ?? Data()
                let parsed = ProviderErrorBody.parse(
                    body,
                    errorTypeHeader: errorTypeHeader.flatMap { result.headers.value(forHTTPHeaderField: $0) }
                )
                let error = ChatError.httpError(
                    statusCode: result.status,
                    message: parsed.message,
                    errorType: parsed.errorType
                )
                let retryable =
                    retryableStatusCodes.contains(result.status) || (retryOnTimeout && result.status == 408)
                if retryable && attempt < maxRetries {
                    delayBeforeRetry = result.status != 408
                    lastError = error
                    continue
                }
                throw error
            } catch let error as ChatError {
                throw error
            } catch is AttemptTimedOut {
                guard retryOnTimeout, attempt < maxRetries else { throw ChatError.timeout }
                lastError = ChatError.timeout
                delayBeforeRetry = false
            } catch let error as URLError where error.code == .timedOut {
                guard retryOnTimeout, attempt < maxRetries else { throw ChatError.timeout }
                lastError = ChatError.timeout
                delayBeforeRetry = false
            } catch {
                try Task.checkCancellation()
                if Self.isCancellation(error) { throw CancellationError() }
                lastError = ChatError.networkError(error.localizedDescription)
                delayBeforeRetry = true
            }
        }

        throw lastError ?? ChatError.networkError("Request failed after \(maxRetries + 1) attempts")
    }

    /// Runs one attempt, failing it with `AttemptTimedOut` once `timeout` passes.
    ///
    /// A shared session can't set a resource timeout per request, and `URLRequest.timeoutInterval`
    /// only limits idle time, so the limit on the whole attempt is enforced here.
    private static func withAttemptTimeout<T: Sendable>(
        _ timeout: TimeInterval,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T?.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(max(0, timeout) * 1_000_000_000))
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let value = first else {
                throw AttemptTimedOut()
            }
            return value
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let error = error as? URLError, error.code == .cancelled { return true }
        return false
    }

    /// Maps errors raised while reading a stream's body, after the retry loop has finished.
    private static func mapTransportError(_ error: Error) -> Error {
        if error is ChatError { return error }
        if isCancellation(error) { return CancellationError() }
        if let error = error as? URLError, error.code == .timedOut { return ChatError.timeout }
        return ChatError.networkError(error.localizedDescription)
    }
}

/// `URLSession.AsyncBytes` isn't `Sendable`, but it's handed from the attempt to the one task
/// that reads it, never shared.
private struct UncheckedBytes: @unchecked Sendable {
    let bytes: URLSession.AsyncBytes
    let response: URLResponse
}
