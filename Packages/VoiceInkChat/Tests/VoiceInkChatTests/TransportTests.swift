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

    @Test func maxRetriesAndCustomStatusesAreRespected() async throws {
        let none = Stub([.respond(status: 500, body: "")])
        await #expect(throws: ChatError.self) {
            try await none.http.send(get(none), timeout: 5, retryOnTimeout: false, maxRetries: 0)
        }
        #expect(none.attempts == 1)

        // Ollama's statuses: 501 isn't retried.
        let ollama = Stub([.respond(status: 501, body: "")])
        await #expect(throws: ChatError.self) {
            try await ollama.http.send(
                get(ollama), timeout: 5, retryOnTimeout: false,
                retryableStatusCodes: ChatHTTP.defaultRetryableStatusCodes)
        }
        #expect(ollama.attempts == 1)
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
        let data = try await stub.http.send(get(stub), timeout: 0.3, retryOnTimeout: true)
        #expect(!data.isEmpty)
        #expect(stub.attempts == 3)
        // Two timed-out attempts and no backoff between them.
        #expect(Date().timeIntervalSince(started) < 2)
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
            if case ChatError.networkError = error { return true }
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

    @Test func oneSharedSessionIsReused() {
        #expect(ChatHTTP.shared.session === ChatHTTP.sharedSession)
        #expect(ChatHTTP().session === ChatHTTP.shared.session)
        #expect(OpenAICompatibleChatClient(apiKey: "k", provider: .openAI).http.session === ChatHTTP.sharedSession)
    }

    // MARK: Streaming

    @Test func streamsServerSentEventsAcrossArbitraryChunks() async throws {
        let stub = Stub([.chunks(
            status: 200,
            headers: ["Content-Type": "text/event-stream"],
            pieces: [": OPENROUTER PROCESSING\n\n", "data: {\"a\":", "1}\n\nevent: done\nda", "ta: [DONE]\r\n\r\n", "data: tail"],
            pause: 0.02
        )])
        var events: [ServerSentEvent] = []
        for try await event in stub.http.events(get(stub), timeout: 5, retryOnTimeout: false) {
            events.append(event)
        }
        #expect(events == [
            ServerSentEvent(data: #"{"a":1}"#),
            ServerSentEvent(event: "done", data: "[DONE]"),
            ServerSentEvent(data: "tail"),
        ])
        #expect(stub.requests.first?.value(forHTTPHeaderField: "Accept") == "text/event-stream")
    }

    @Test func streamRetriesBeforeHeadersAndReportsHTTPErrors() async throws {
        let stub = Stub([.respond(status: 503, body: "busy"), .chunks(status: 200, pieces: ["data: hi\n\n"])])
        var events: [ServerSentEvent] = []
        for try await event in stub.http.events(get(stub), timeout: 5, retryOnTimeout: false) {
            events.append(event)
        }
        #expect(events.map(\.data) == ["hi"])
        #expect(stub.attempts == 2)

        let failing = Stub([.respond(status: 400, body: #"{"error": {"message": "stream not allowed"}}"#)])
        await #expect(throws: ChatError.httpError(statusCode: 400, message: "stream not allowed", errorType: nil)) {
            for try await _ in failing.http.events(get(failing), timeout: 5, retryOnTimeout: false) {}
        }
    }

    // MARK: Clients end to end

    @Test func clientsSendThroughTheTransport() async throws {
        let stub = Stub([
            .respond(status: 500, body: "oops"),
            .respond(status: 200, body: #"{"choices": [{"message": {"content": "Done."}, "finish_reason": "stop"}]}"#),
        ])
        let client = OpenAICompatibleChatClient(
            apiKey: "k",
            provider: .custom(endpoint: stub.url("v1/chat/completions")),
            http: stub.http
        )
        let result = try await client.complete(dictation)
        #expect(result.text == "Done.")
        #expect(stub.attempts == 2)
        #expect(try stub.lastBodyJSON()["model"] == "test-model")
    }

    @Test func ollamaClientUsesItsBaseURL() async throws {
        let stub = Stub([.respond(status: 200, body: #"{"message": {"content": "Hi"}, "done_reason": "stop"}"#)])
        let result = try await OllamaChatClient(baseURL: stub.baseURL, http: stub.http).complete(dictation)
        #expect(result.text == "Hi")
        #expect(stub.requests.first?.url?.path == "/api/chat")
    }

    @Test func geminiFallsBackToV1BetaOn404AndRemembersIt() async throws {
        let model = "gemini-test-\(UUID().uuidString)"
        let answer = #"{"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}]}"#
        let stub = Stub([
            .respond(status: 404, body: #"{"error": {"code": 404, "message": "models/x is not found for API version v1", "status": "NOT_FOUND"}}"#),
            .respond(status: 200, body: answer),
        ])
        let client = GeminiChatClient(apiKey: "k", baseURL: stub.baseURL, http: stub.http)
        var chat = dictation
        chat.model = model
        #expect(try await client.complete(chat).text == "Hi")
        #expect(try await client.complete(chat).text == "Hi")
        #expect(stub.requests.map { $0.url?.path } == ["/v1/interactions", "/v1beta/interactions", "/v1beta/interactions"])

        // An explicit version is used as given.
        let explicit = Stub([.respond(status: 200, body: answer)])
        let pinned = GeminiChatClient(apiKey: "k", apiVersion: .v1beta, baseURL: explicit.baseURL, http: explicit.http)
        _ = try await pinned.complete(dictation)
        #expect(explicit.requests.map { $0.url?.path } == ["/v1beta/interactions"])
    }
}
