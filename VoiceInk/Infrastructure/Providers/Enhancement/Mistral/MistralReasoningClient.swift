import Foundation
import LLMkit

struct MistralChatResult: Equatable, Sendable {
    /// The final answer: the plain `content` string, or every top-level `text` chunk joined.
    /// `thinking` chunks are never included.
    let text: String
    let finishReason: String?

    var wasTruncated: Bool {
        finishReason == "length" || finishReason == "model_length"
    }
}

/// Chat completions client for Mistral requests with a picked reasoning level.
///
/// With `reasoning_effort` on, Mistral returns `message.content` as a list of chunks, where
/// `thinking` chunks hold the reasoning and `text` chunks hold the answer. LLMkit's
/// `OpenAILLMClient` decodes `content` as a string, so these requests can't use it. Default
/// requests still go through LLMkit unchanged.
///
/// The body is LLMkit's (`model`, `messages`, `temperature: 0.3`, `stream: false`) plus
/// `reasoning_effort`.
enum MistralReasoningClient {
    static let endpoint = URL(string: "https://api.mistral.ai/v1/chat/completions")!

    static func chatCompletion(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        reasoningEffort: String,
        timeout: TimeInterval,
        retryOnTimeout: Bool
    ) async throws -> MistralChatResult {
        let request = try makeRequest(
            apiKey: apiKey,
            model: model,
            messages: messages,
            systemPrompt: systemPrompt,
            reasoningEffort: reasoningEffort
        )
        let data = try await ReasoningChatHTTP.send(request, timeout: timeout, retryOnTimeout: retryOnTimeout)
        return try decodeResult(from: data)
    }

    static func makeRequest(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        reasoningEffort: String
    ) throws -> URLRequest {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.missingAPIKey
        }

        var allMessages = messages
        if let systemPrompt, !systemPrompt.isEmpty {
            allMessages.insert(.system(systemPrompt), at: 0)
        }

        let body: [String: Any] = [
            "model": model,
            "messages": allMessages.map { ["role": $0.role, "content": $0.content] },
            "temperature": 0.3,
            "stream": false,
            "reasoning_effort": reasoningEffort,
        ]
        return try ReasoningChatHTTP.jsonRequest(
            url: endpoint,
            body: body,
            headers: ["Authorization": "Bearer \(apiKey)"]
        )
    }

    static func decodeResult(from data: Data) throws -> MistralChatResult {
        let decoded: ResponseBody
        do {
            decoded = try JSONDecoder().decode(ResponseBody.self, from: data)
        } catch {
            throw LLMKitError.decodingError(error.localizedDescription)
        }
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

        /// Only top-level `text` chunks. A `thinking` chunk nests its own text chunks under
        /// `thinking`, which isn't decoded, so reasoning can't reach the answer.
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
