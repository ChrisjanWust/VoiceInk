import Foundation
import Testing
@testable import VoiceInkChat

/// ChatHTTP's retry, timeout and error behaviour, against a stub URLProtocol. Backoff is shortened
/// to 10 ms; the policy (which failures retry, how often, and whether they wait) is LLMkit's.
@Suite struct TransportTests {
    private func get(_ stub: Stub) -> URLRequest {
        var request = URLRequest(url: stub.url("v1/test"))
        request.httpMethod = "POST"
        return request
    }

    private let ok = StubReply.respond(status: 200, body: #"{"ok": true}"#)

    @Test func retriesRateLimitsAndServerErrorsThenSucceeds() async throws {
        let stub = Stub([.respond(status: 429, body: "slow down"), .respond(status: 503, body: "busy"), ok])
        let data = try await stub.http.send(get(stub), timeout: 5, retryOnTimeout: false)
        #expect(String(data: data, encoding: .utf8) == #"{"ok": true}"#)
        #expect(stub.attempts == 3)
    }

    @Test func givesUpAfterThreeAttempts() async throws {
        let stub = Stub([.respond(status: 529, body: #"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#)])
        await #expect(throws: ChatError.httpError(statusCode: 529, message: "Overloaded", errorType: "overloaded_error")) {
            try await stub.http.send(get(stub), timeout: 5, retryOnTimeout: false)
        }
        #expect(stub.attempts == 3)
    }

    @Test func clientErrorsAreNotRetried() async throws {
        let stub = Stub([.respond(status: 401, body: #"{"error": {"message": "Bad key", "type": "auth"}}"#)])
        await #expect(throws: ChatError.httpError(statusCode: 401, message: "Bad key", errorType: "auth")) {
            try await stub.http.send(get(stub), timeout: 5, retryOnTimeout: true)
        }
        #expect(stub.attempts == 1)
    }

    @Test func ollamaStatusesDontRetry501() async throws {
        let ollama = Stub([.respond(status: 501, body: "")])
        await #expect(throws: ChatError.self) {
            try await ollama.http.send(
                get(ollama), timeout: 5, retryOnTimeout: false,
                retryableStatusCodes: ChatHTTP.defaultRetryableStatusCodes)
        }
        #expect(ollama.attempts == 1)
    }

    @Test func serverErrorsWaitBeforeRetrying() async throws {
        let stub = Stub([.respond(status: 503, body: "busy"), ok])
        let started = Date()
        _ = try await stub.http(retryBaseDelay: 0.5).send(get(stub), timeout: 5, retryOnTimeout: false)
        #expect(stub.attempts == 2)
        #expect(Date().timeIntervalSince(started) >= 0.5)
    }

    @Test func timeoutWithoutRetryFailsAfterOneAttempt() async throws {
        let stub = Stub([.hang])
        let started = Date()
        await #expect(throws: ChatError.timeout) {
            try await stub.http.send(get(stub), timeout: 0.3, retryOnTimeout: false)
        }
        #expect(stub.attempts == 1)
        #expect(Date().timeIntervalSince(started) < 2)
    }

    @Test func timeoutIsPerAttemptAndRetriedImmediatelyWhenAllowed() async throws {
        let stub = Stub([.hang, .hang, ok])
        let started = Date()
        // With a 0.5 s backoff, waiting before either timeout retry would take at least 1.1 s.
        let data = try await stub.http(retryBaseDelay: 0.5).send(get(stub), timeout: 0.3, retryOnTimeout: true)
        #expect(!data.isEmpty)
        #expect(stub.attempts == 3)
        #expect(Date().timeIntervalSince(started) < 1.05)
    }

    @Test func aTimedOutAttemptGetsANewSession() async throws {
        let stub = Stub([.hang, ok])
        let http = stub.http
        let before = http.sessions.current
        _ = try await http.send(get(stub), timeout: 0.3, retryOnTimeout: true)
        let afterTimeout = http.sessions.current
        #expect(afterTimeout !== before)

        // A successful request keeps the session, so the connection is reused.
        let reused = Stub([ok])
        let reusedHTTP = reused.http
        let first = reusedHTTP.sessions.current
        _ = try await reusedHTTP.send(get(reused), timeout: 5, retryOnTimeout: false)
        #expect(reusedHTTP.sessions.current === first)
    }

    @Test func slowResponseCountsAgainstTheWholeAttempt() async throws {
        // The server answers after 0.6 s; the attempt only has 0.25 s.
        let stub = Stub([.respond(status: 200, body: "{}", delay: 0.6)])
        await #expect(throws: ChatError.timeout) {
            try await stub.http.send(get(stub), timeout: 0.25, retryOnTimeout: false)
        }
    }

    @Test func urlSessionTimeoutsMapToTimeout() async throws {
        let stub = Stub([.fail(.timedOut), .fail(.timedOut), .fail(.timedOut)])
        await #expect(throws: ChatError.timeout) {
            try await stub.http.send(get(stub), timeout: 5, retryOnTimeout: true)
        }
        #expect(stub.attempts == 3)
    }

    @Test func http408RetriesOnlyWithRetryOnTimeout() async throws {
        let retried = Stub([.respond(status: 408, body: ""), ok])
        _ = try await retried.http.send(get(retried), timeout: 5, retryOnTimeout: true)
        #expect(retried.attempts == 2)

        let notRetried = Stub([.respond(status: 408, body: "Request Timeout")])
        await #expect(throws: ChatError.httpError(statusCode: 408, message: "Request Timeout", errorType: nil)) {
            try await notRetried.http.send(get(notRetried), timeout: 5, retryOnTimeout: false)
        }
        #expect(notRetried.attempts == 1)
    }

    @Test func networkErrorsAreRetriedThenReported() async throws {
        let recovers = Stub([.fail(.networkConnectionLost), ok])
        _ = try await recovers.http.send(get(recovers), timeout: 5, retryOnTimeout: false)
        #expect(recovers.attempts == 2)

        let fails = Stub([.fail(.cannotFindHost)])
        await #expect {
            try await fails.http.send(get(fails), timeout: 5, retryOnTimeout: false)
        } throws: { error in
            // The URLError code survives, so the app can name an unknown host.
            if case ChatError.networkError(_, let code) = error { return code == .cannotFindHost }
            return false
        }
        #expect(fails.attempts == 3)
    }

    @Test func cancellationPassesThrough() async throws {
        let stub = Stub([.hang])
        let http = stub.http
        let request = get(stub)
        let task = Task { try await http.send(request, timeout: 10, retryOnTimeout: true) }
        try await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        let result = await task.result
        #expect(throws: CancellationError.self) { try result.get() }
        #expect(stub.attempts == 1)
    }

    @Test func bedrockErrorTypeComesFromTheHeader() async throws {
        let stub = Stub([.respond(
            status: 403,
            headers: ["x-amzn-ErrorType": "AccessDeniedException:http://internal.amazon.com/coral/"],
            body: #"{"message": "You don't have access to the model."}"#
        )])
        await #expect(throws: ChatError.httpError(
            statusCode: 403, message: "You don't have access to the model.", errorType: "AccessDeniedException")) {
            try await stub.http.send(get(stub), timeout: 5, retryOnTimeout: false, errorTypeHeader: "x-amzn-ErrorType")
        }
    }

    // MARK: Clients end to end

    @Test func clientsSendThroughTheTransport() async throws {
        let stub = Stub([
            .respond(status: 500, body: "oops"),
            .respond(status: 200, body: #"{"choices": [{"message": {"content": "Done."}, "finish_reason": "stop"}]}"#),
        ])
        var client = OpenAICompatibleChatClient(apiKey: "k", provider: .custom(endpoint: stub.url("v1/chat/completions")))
        client.http = stub.http
        let result = try await client.complete(dictation)
        #expect(result.text == "Done.")
        #expect(stub.attempts == 2)
        #expect(try stub.lastBodyJSON()["model"] == "test-model")
    }

    @Test func ollamaClientUsesItsBaseURL() async throws {
        let stub = Stub([.respond(status: 200, body: #"{"message": {"content": "Hi"}, "done_reason": "stop"}"#)])
        var client = OllamaChatClient(baseURL: stub.baseURL)
        client.http = stub.http
        let result = try await client.complete(dictation)
        #expect(result.text == "Hi")
        #expect(stub.requests.first?.url?.path == "/api/chat")
    }

    @Test func geminiSendsToTheVersionItWasGiven() async throws {
        let answer = #"{"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}]}"#
        let stub = Stub([.respond(status: 200, body: answer)])
        var client = GeminiChatClient(apiKey: "k", apiVersion: .v1beta)
        client.baseURL = stub.baseURL
        client.http = stub.http
        #expect(try await client.complete(dictation).text == "Hi")
        #expect(stub.requests.map { $0.url?.path } == ["/v1beta/interactions"])
    }
}
