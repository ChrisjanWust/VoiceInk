import Foundation

/// A client that turns one chat request into one final answer.
///
/// Every provider client in this package conforms. Clients are value types that hold their
/// configuration (API key, endpoint) and share one transport, so they're cheap to create per
/// request.
public protocol ChatClient: Sendable {
    func complete(_ request: ChatRequest) async throws -> ChatResult
}

/// One chat request: text in, text out.
public struct ChatRequest: Sendable, Equatable {
    public struct Message: Sendable, Equatable {
        public enum Role: String, Sendable {
            case system
            case user
            case assistant
        }

        public var role: Role
        public var content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }

        public static func system(_ content: String) -> Message { Message(role: .system, content: content) }
        public static func user(_ content: String) -> Message { Message(role: .user, content: content) }
        public static func assistant(_ content: String) -> Message { Message(role: .assistant, content: content) }
    }

    /// How much the model should reason. Each client maps it to its provider's field, and throws
    /// ``ChatError/unsupported(_:)`` for a setting its API has no way to express.
    ///
    /// | Case | OpenAI-compatible | Anthropic | Gemini | OpenRouter | Ollama |
    /// |---|---|---|---|---|---|
    /// | `.effort(e)` | `reasoning_effort` | adaptive thinking + `output_config.effort` | `thinking_level` | `reasoning.effort` | `think: e` |
    /// | `.disabled` | `reasoning_effort: "none"` | `thinking.type: disabled` | unsupported | `reasoning.enabled: false` | `think: false` |
    /// | `.enabled` | unsupported | adaptive thinking | unsupported | `reasoning.enabled: true` | `think: true` |
    ///
    /// The app picks what each model takes. OpenRouter always adds `exclude: true`, so reasoning
    /// text isn't returned. Anything else, such as an Anthropic token budget or `between_tools`,
    /// goes in ``extraBody``. Bedrock takes reasoning per model in `additionalModelRequestFields`,
    /// so it throws for every case. Leave it `nil` to send no reasoning field at all.
    public enum Reasoning: Sendable, Equatable {
        /// A named level, passed through as written: `"none"`, `"minimal"`, `"low"`, `"medium"`,
        /// `"high"`, `"xhigh"`, `"max"`, or whatever else the model accepts.
        case effort(String)
        /// Turn reasoning off.
        case disabled
        /// Turn reasoning on at the provider's default amount.
        case enabled
    }

    public var model: String
    /// The system prompt. When set, it replaces any `.system` messages for providers that take
    /// the system prompt as a separate field (Anthropic, Gemini, Bedrock). OpenAI-compatible
    /// providers get it as a leading system message.
    public var system: String?
    public var messages: [Message]
    /// The most tokens to generate, including any thinking. `nil` sends the provider's own
    /// default, except Anthropic, which requires the field and gets 8192.
    public var maxTokens: Int?
    /// Sampling temperature. `nil` leaves the field out of the request entirely.
    public var temperature: Double?
    public var reasoning: Reasoning?
    /// Top-level body fields added last, so they override anything the client set. This is
    /// where Fast mode's fields go, such as `service_tier` or `speed`. `.null` sends an explicit
    /// `null`.
    public var extraBody: [String: JSONValue]
    /// Headers added on top of the client's own, such as a beta header. Authentication headers
    /// (`Authorization`, `x-api-key`, `anthropic-version`, `x-goog-api-key`) always come from the
    /// client, as in the app's in-app clients, so an extra header can't replace the key.
    public var extraHeaders: [String: String]
    /// The timeout for each attempt, in seconds. Up to three attempts are made.
    public var timeout: TimeInterval
    /// Whether a timed-out attempt (or HTTP 408) is retried, immediately and within the same
    /// three-attempt budget.
    public var retryOnTimeout: Bool

    public init(
        model: String,
        system: String? = nil,
        messages: [Message],
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        reasoning: Reasoning? = nil,
        extraBody: [String: JSONValue] = [:],
        extraHeaders: [String: String] = [:],
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) {
        self.model = model
        self.system = system
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.extraBody = extraBody
        self.extraHeaders = extraHeaders
        self.timeout = timeout
        self.retryOnTimeout = retryOnTimeout
    }

    /// The system prompt for APIs with a separate system field: `system` if it's set and not
    /// blank, otherwise every `.system` message joined with newlines.
    var resolvedSystem: String? {
        if let system, !system.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return system
        }
        let systemMessages = messages.filter { $0.role == .system }.map(\.content)
        return systemMessages.isEmpty ? nil : systemMessages.joined(separator: "\n")
    }

    /// The user and assistant turns, without system messages.
    var conversation: [Message] {
        messages.filter { $0.role != .system }
    }
}

/// Why the model stopped.
public enum FinishReason: String, Sendable, Equatable {
    /// It finished its answer.
    case stop
    /// It hit the output token limit (or the context window), so the answer is cut off.
    case length
    /// A safety system or guardrail stopped or replaced the answer.
    case contentFilter
    /// It stopped to call a tool. VoiceInk sends no tools, so this is unexpected.
    case toolCalls
    /// The provider reported a failure.
    case error
    /// The provider didn't say, or said something this library doesn't map.
    case other
}

public struct ChatUsage: Sendable, Equatable {
    public var inputTokens: Int?
    public var outputTokens: Int?
    /// Reasoning tokens, when the provider reports them separately.
    public var reasoningTokens: Int?

    init(inputTokens: Int? = nil, outputTokens: Int? = nil, reasoningTokens: Int? = nil) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.reasoningTokens = reasoningTokens
    }

    var isEmpty: Bool { inputTokens == nil && outputTokens == nil && reasoningTokens == nil }
}

/// The model's final answer.
///
/// Check ``finishReason`` before using ``text``: only `.stop` means a complete answer. Text with
/// `.length` is cut off, text with `.contentFilter` may be a guardrail's replacement, and
/// `.error`, `.toolCalls` or `.other` can come with a partial answer.
public struct ChatResult: Sendable, Equatable {
    /// The answer, trimmed of surrounding whitespace. Never contains reasoning: thinking blocks,
    /// `reasoning_content`, reasoning chunks and `<think>` sections are all left out.
    public var text: String
    public var finishReason: FinishReason
    /// The provider's own finish value, such as `"max_tokens"` or `"incomplete"`.
    public var rawFinishReason: String?
    public var usage: ChatUsage?
    /// The upstream provider that served the request (OpenRouter).
    public var provider: String?
    /// Provider metadata passed through as JSON, such as OpenRouter's `openrouter_metadata`.
    public var metadata: JSONValue?

    init(
        text: String,
        finishReason: FinishReason,
        rawFinishReason: String? = nil,
        usage: ChatUsage? = nil,
        provider: String? = nil,
        metadata: JSONValue? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.rawFinishReason = rawFinishReason
        self.usage = usage
        self.provider = provider
        self.metadata = metadata
    }

    /// Whether the answer was cut off by a token limit, so it shouldn't be pasted as-is.
    public var isTruncated: Bool { finishReason == .length }

    /// Builds a result, trimming the text, and throws ``ChatError/noResultReturned`` when there's
    /// no text and nothing explains why. An empty answer that was truncated or filtered is
    /// returned, so the caller can report the real reason.
    static func make(
        text: String,
        finishReason: FinishReason,
        rawFinishReason: String?,
        usage: ChatUsage?,
        provider: String? = nil,
        metadata: JSONValue? = nil
    ) throws -> ChatResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty, finishReason != .length, finishReason != .contentFilter {
            throw ChatError.noResultReturned
        }
        return ChatResult(
            text: trimmed,
            finishReason: finishReason,
            rawFinishReason: rawFinishReason,
            usage: usage.flatMap { $0.isEmpty ? nil : $0 },
            provider: provider,
            metadata: metadata
        )
    }
}
