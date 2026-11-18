import Foundation

/// The HTTP transport every chat client shares.
///
/// It keeps the retry and timeout rules of LLMkit's chat clients and of the app's in-app
/// `ChatHTTP`, so switching a call site over doesn't change how failures behave:
/// - up to three attempts;
/// - a 1 s then 2 s backoff for retryable statuses (429 and every 5xx by default) and for
///   network errors;
/// - timeouts, including HTTP 408, retried immediately and only when `retryOnTimeout` is on;
/// - the timeout limits each whole attempt, not just idle time;
/// - any other non-2xx status throws ``ChatError/httpError(statusCode:message:errorType:)``;
/// - cancellation passes straight through as `CancellationError`.
///
/// One difference, on purpose: LLMkit makes a new ephemeral `URLSession` for every attempt.
/// ``shared`` reuses one session, so requests to the same host keep the connection alive and
/// skip a fresh TCP and TLS handshake. A connection can stall, though (after sleep, a network or
/// VPN change, or a bad edge node), so after a timeout or network error the session is
/// replaced: the retry, and every later request, start on a new connection, as with LLMkit.
struct ChatHTTP: Sendable {
    /// LLMkit's chat retry statuses: 429 and every 5xx.
    static let chatRetryableStatusCodes: Set<Int> = Set([429]).union(500...599)
    /// LLMkit's default retry statuses, which its Ollama client uses.
    static let defaultRetryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]
    /// LLMkit's attempt budget: the first try and two retries.
    static let maxRetries = 2

    /// One transport for the whole process, so connections are reused across requests.
    static let shared = ChatHTTP(makeConfiguration: { .ephemeral }, retryBaseDelay: 1)

    let sessions: SessionHolder
    /// The first retry waits this long, and the second twice as long. Tests shorten it.
    let retryBaseDelay: TimeInterval

    /// - Parameter makeConfiguration: The configuration for each session, before the chat
    ///   settings are applied. Tests add a stub `URLProtocol` here.
    init(makeConfiguration: @escaping @Sendable () -> URLSessionConfiguration, retryBaseDelay: TimeInterval) {
        self.sessions = SessionHolder(makeConfiguration: makeConfiguration)
        self.retryBaseDelay = retryBaseDelay
    }

    // MARK: - Building requests

    /// A POST with a JSON body. Keys are sorted, so the same body always encodes the same way.
    static func jsonRequest(url: URL, body: [String: JSONValue], headers: [String: String]) throws -> URLRequest {
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

    /// Merges header layers, with each later layer winning. Names compare case-insensitively, as
    /// in HTTP and `URLRequest`, so `authorization` in one layer replaces `Authorization` in
    /// another instead of racing it.
    static func mergedHeaders(_ layers: [String: String]...) -> [String: String] {
        var byName: [String: (name: String, value: String)] = [:]
        for layer in layers {
            for (name, value) in layer {
                byName[name.lowercased()] = (name, value)
            }
        }
        return Dictionary(uniqueKeysWithValues: byName.values.map { ($0.name, $0.value) })
    }

    /// Decodes a response body, reporting failures as ``ChatError/decodingError(_:)``.
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw ChatError.decodingError(error.localizedDescription)
        }
    }

    // MARK: - Sending

    /// Sends the request and returns the body of a 2xx response.
    ///
    /// - Parameter errorTypeHeader: A response header that names the error type, such as
    ///   Bedrock's `x-amzn-ErrorType`. It wins over a type found in the body.
    func send(
        _ request: URLRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool,
        retryableStatusCodes: Set<Int> = ChatHTTP.chatRetryableStatusCodes,
        errorTypeHeader: String? = nil
    ) async throws -> Data {
        var request = request
        request.timeoutInterval = timeout
        var lastError: Error?
        var delayBeforeRetry = true

        for attempt in 0...Self.maxRetries {
            try Task.checkCancellation()
            if attempt > 0 && delayBeforeRetry {
                let seconds = retryBaseDelay * pow(2.0, Double(attempt - 1))
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            }

            let session = sessions.current
            do {
                let (data, response) = try await Self.data(for: request, on: session, timeout: timeout)
                guard let http = response as? HTTPURLResponse else {
                    throw ChatError.networkError("No HTTP response received.", code: nil)
                }
                if (200..<300).contains(http.statusCode) {
                    return data
                }
                let parsed = ProviderErrorBody.parse(
                    data,
                    errorTypeHeader: errorTypeHeader.flatMap { http.value(forHTTPHeaderField: $0) }
                )
                let error = ChatError.httpError(
                    statusCode: http.statusCode,
                    message: parsed.message,
                    errorType: parsed.errorType
                )
                let retryable =
                    retryableStatusCodes.contains(http.statusCode) || (retryOnTimeout && http.statusCode == 408)
                if retryable && attempt < Self.maxRetries {
                    delayBeforeRetry = http.statusCode != 408
                    lastError = error
                    continue
                }
                throw error
            } catch let error as ChatError {
                throw error
            } catch is AttemptTimedOut {
                sessions.replace(session)
                guard retryOnTimeout, attempt < Self.maxRetries else { throw ChatError.timeout }
                lastError = ChatError.timeout
                delayBeforeRetry = false
            } catch let error as URLError where error.code == .timedOut {
                sessions.replace(session)
                guard retryOnTimeout, attempt < Self.maxRetries else { throw ChatError.timeout }
                lastError = ChatError.timeout
                delayBeforeRetry = false
            } catch {
                try Task.checkCancellation()
                if error is CancellationError || (error as? URLError)?.code == .cancelled {
                    throw CancellationError()
                }
                sessions.replace(session)
                lastError = ChatError.networkError(error.localizedDescription, code: (error as? URLError)?.code)
                delayBeforeRetry = true
            }
        }

        throw lastError ?? ChatError.networkError("Request failed after \(Self.maxRetries + 1) attempts", code: nil)
    }

    private struct AttemptTimedOut: Error {}

    /// Runs one attempt, failing it with `AttemptTimedOut` once `timeout` passes.
    ///
    /// A shared session can't set a resource timeout per request, and `URLRequest.timeoutInterval`
    /// only limits idle time, so the limit on the whole attempt is enforced here.
    private static func data(
        for request: URLRequest,
        on session: URLSession,
        timeout: TimeInterval
    ) async throws -> (Data, URLResponse) {
        // Clamped, so an infinite or huge timeout can't overflow the conversion to nanoseconds.
        let nanoseconds = UInt64(min(max(0, timeout), 86_400) * 1_000_000_000)
        return try await withThrowingTaskGroup(of: (Data, URLResponse)?.self) { group in
            group.addTask { try await session.data(for: request) }
            group.addTask {
                try await Task.sleep(nanoseconds: nanoseconds)
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let result = first else {
                throw AttemptTimedOut()
            }
            return result
        }
    }
}

/// The session ``ChatHTTP`` sends on, replaced when a connection may have stalled.
final class SessionHolder: @unchecked Sendable {
    private let lock = NSLock()
    private let makeConfiguration: @Sendable () -> URLSessionConfiguration
    private var session: URLSession

    init(makeConfiguration: @escaping @Sendable () -> URLSessionConfiguration) {
        self.makeConfiguration = makeConfiguration
        self.session = Self.makeSession(makeConfiguration())
    }

    var current: URLSession { lock.withLock { session } }

    /// Replaces `stale` with a new session, unless another request already has.
    ///
    /// The old session isn't invalidated: another request may have just read it and not yet
    /// started its task, and starting a task on an invalidated session raises an uncatchable
    /// exception. With no delegate, the old session is released once its requests finish.
    func replace(_ stale: URLSession) {
        lock.withLock {
            guard session === stale else { return }
            session = Self.makeSession(makeConfiguration())
        }
    }

    /// Ephemeral (nothing written to disk), no cache, no cookies. ``ChatHTTP`` enforces the
    /// per-attempt timeout, so the session's own resource timeout is only a backstop.
    private static func makeSession(_ configuration: URLSessionConfiguration) -> URLSession {
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        configuration.httpShouldSetCookies = false
        configuration.httpCookieAcceptPolicy = .never
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForResource = 600
        return URLSession(configuration: configuration)
    }
}
