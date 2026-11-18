import Foundation
@testable import VoiceInkChat

/// What a stubbed endpoint does for one attempt.
enum StubReply: Sendable {
    case respond(status: Int, headers: [String: String] = [:], body: String, delay: TimeInterval = 0)
    /// Sends the headers, then the body in pieces with a pause between them.
    case chunks(status: Int, headers: [String: String] = [:], pieces: [String], pause: TimeInterval = 0)
    /// Never answers, until the client gives up.
    case hang
    case fail(URLError.Code)
}

/// A stubbed host. Requests are routed by host name, so tests running in parallel don't share
/// stubs.
final class Stub: @unchecked Sendable {
    let host: String
    private let lock = NSLock()
    private var replies: [StubReply]
    private var recorded: [URLRequest] = []
    private var bodies: [Data] = []

    /// `replies` are used in order, and the last one repeats.
    init(_ replies: [StubReply]) {
        self.host = "stub-\(UUID().uuidString.lowercased()).test"
        self.replies = replies
        StubRegistry.shared.register(self)
    }

    var baseURL: URL { URL(string: "https://\(host)")! }

    func url(_ path: String) -> URL { baseURL.appendingPathComponent(path) }

    var requests: [URLRequest] { lock.withLock { recorded } }
    var requestBodies: [Data] { lock.withLock { bodies } }
    var attempts: Int { requests.count }

    func lastBodyJSON() throws -> JSONValue {
        try JSONValue(data: requestBodies.last ?? Data())
    }

    fileprivate func nextReply(for request: URLRequest, body: Data) -> StubReply {
        lock.withLock {
            recorded.append(request)
            bodies.append(body)
            return replies.count > 1 ? replies.removeFirst() : replies[0]
        }
    }

    /// A transport whose retries wait 10 ms instead of 1 s and 2 s.
    var http: ChatHTTP { http(retryBaseDelay: 0.01) }

    func http(retryBaseDelay: TimeInterval) -> ChatHTTP {
        ChatHTTP(
            makeConfiguration: {
                let configuration = URLSessionConfiguration.ephemeral
                configuration.protocolClasses = [StubURLProtocol.self]
                return configuration
            },
            retryBaseDelay: retryBaseDelay
        )
    }
}

final class StubRegistry: @unchecked Sendable {
    static let shared = StubRegistry()
    private let lock = NSLock()
    private var stubs: [String: Stub] = [:]

    func register(_ stub: Stub) { lock.withLock { stubs[stub.host] = stub } }
    func stub(for host: String?) -> Stub? { lock.withLock { host.flatMap { stubs[$0] } } }
}

final class StubURLProtocol: URLProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    override class func canInit(with request: URLRequest) -> Bool {
        StubRegistry.shared.stub(for: request.url?.host) != nil
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let stub = StubRegistry.shared.stub(for: request.url?.host) else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        let reply = stub.nextReply(for: request, body: Self.body(of: request))
        switch reply {
        case .hang:
            break
        case .fail(let code):
            client?.urlProtocol(self, didFailWithError: URLError(code))
        case .respond(let status, let headers, let body, let delay):
            after(delay) { [self] in
                send(status: status, headers: headers)
                client?.urlProtocol(self, didLoad: Data(body.utf8))
                client?.urlProtocolDidFinishLoading(self)
            }
        case .chunks(let status, let headers, let pieces, let pause):
            send(status: status, headers: headers)
            sendPieces(pieces[...], pause: pause)
        }
    }

    override func stopLoading() {
        lock.withLock { stopped = true }
    }

    private var isStopped: Bool { lock.withLock { stopped } }

    private func send(status: Int, headers: [String: String]) {
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    }

    private func sendPieces(_ pieces: ArraySlice<String>, pause: TimeInterval) {
        guard !isStopped else { return }
        guard let piece = pieces.first else {
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        client?.urlProtocol(self, didLoad: Data(piece.utf8))
        after(pause) { [self] in sendPieces(pieces.dropFirst(), pause: pause) }
    }

    private func after(_ delay: TimeInterval, _ work: @escaping @Sendable () -> Void) {
        let run: @Sendable () -> Void = { [self] in if !isStopped { work() } }
        if delay <= 0 {
            run()
        } else {
            DispatchQueue.global().asyncAfter(deadline: .now() + delay, execute: run)
        }
    }

    /// URLSession hands protocols the body as a stream, not `httpBody`.
    private static func body(of request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }
}

// MARK: - Helpers

extension JSONValue {
    /// Parses a JSON literal in a test.
    init(json: String) {
        self = try! JSONValue(data: Data(json.utf8))
    }

    var prettyString: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        return String(data: try! encoder.encode(self), encoding: .utf8)!
    }
}

func bodyJSON(_ request: URLRequest) throws -> JSONValue {
    try JSONValue(data: request.httpBody ?? Data())
}

let dictation = ChatRequest(
    model: "test-model",
    system: "Clean up the transcript. Reply with the cleaned text only.",
    messages: [.user("um so the meeting is uh moved to thursday")]
)
