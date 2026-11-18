import Foundation
import LLMkit

/// One Mistral chat completions request. Optional fields are left out of the body when `nil`.
struct MistralChatRequest {
    var model: String
    /// All turns, with any system prompt first.
    var messages: [ChatMessage]
    /// LLMkit's default for OpenAI-compatible providers.
    var temperature: Double? = 0.3
    /// `reasoning_effort`, such as `"none"` or `"high"`.
    var reasoningEffort: String?

    /// Builds a request the way LLMkit's `OpenAILLMClient` does: a non-empty system prompt
    /// goes first as a system message.
    init(model: String, messages: [ChatMessage], systemPrompt: String?) {
        self.model = model
        var allMessages = messages
        if let systemPrompt, !systemPrompt.isEmpty {
            allMessages.insert(.system(systemPrompt), at: 0)
        }
        self.messages = allMessages
    }

    var jsonBody: [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "stream": false,
        ]
        if let temperature { body["temperature"] = temperature }
        if let reasoningEffort { body["reasoning_effort"] = reasoningEffort }
        return body
    }
}

struct MistralChatResult: Equatable, Sendable {
    /// The final answer: the plain `content` string, or every top-level `text` chunk joined.
    /// `thinking` chunks are never included.
    let text: String
    let finishReason: String?

    var wasTruncated: Bool {
        finishReason == "length" || finishReason == "model_length"
    }
}

/// Client for Mistral's chat completions API (`POST https://api.mistral.ai/v1/chat/completions`).
///
/// Mistral's reasoning models return `message.content` as a list of chunks, where `thinking`
/// chunks hold the reasoning and `text` chunks hold the answer. LLMkit's `OpenAILLMClient`
/// decodes `content` only as a string, so VoiceInk uses this client for those requests.
enum MistralChatClient {
    static let endpoint = URL(string: "https://api.mistral.ai/v1/chat/completions")!

    static func chatCompletion(
        apiKey: String,
        request: MistralChatRequest,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> MistralChatResult {
        let urlRequest = try makeURLRequest(apiKey: apiKey, request: request)
        let data = try await ChatHTTP.send(urlRequest, timeout: timeout, retryOnTimeout: retryOnTimeout)
        return try decodeResult(from: data)
    }

    static func makeURLRequest(apiKey: String, request: MistralChatRequest) throws -> URLRequest {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.missingAPIKey
        }
        return try ChatHTTP.jsonRequest(
            url: endpoint,
            body: request.jsonBody,
            headers: ["Authorization": "Bearer \(apiKey)"]
        )
    }

    static func decodeResult(from data: Data) throws -> MistralChatResult {
        let decoded = try ChatHTTP.decode(ResponseBody.self, from: data)
        guard let choice = decoded.choices.first else {
            return MistralChatResult(text: "", finishReason: nil)
        }
        return MistralChatResult(text: choice.message.content?.answer ?? "", finishReason: choice.finishReason)
    }

    private struct ResponseBody: Decodable {
        struct Choice: Decodable {
            let message: Message
            let finishReason: String?

            enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }

        struct Message: Decodable {
            let content: Content?
        }

        let choices: [Choice]
    }

    /// `content` is a string without reasoning, and a list of chunks with it.
    private enum Content: Decodable {
        case text(String)
        case chunks([Chunk])

        /// Only top-level `text` chunks are read. A `thinking` chunk nests its own text
        /// chunks under `thinking`, which isn't decoded, so reasoning can't reach the answer.
        struct Chunk: Decodable {
            let type: String
            let text: String?
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else {
                self = .chunks(try container.decode([Chunk].self))
            }
        }

        var answer: String {
            switch self {
            case .text(let text):
                return text
            case .chunks(let chunks):
                return chunks.filter { $0.type == "text" }.compactMap(\.text).joined()
            }
        }
    }
}
