import Foundation

/// Fetches the live model lists for providers whose static presets go stale, and keeps
/// only the text models that VoiceInk can use for enhancement.
enum ProviderModelCatalog {
    enum CatalogError: LocalizedError {
        case missingAPIKey
        case httpError(statusCode: Int, message: String)
        case invalidResponse
        case noUsableModels

        var errorDescription: String? {
            switch self {
            case .missingAPIKey:
                return "API key is missing or empty."
            case .httpError(let statusCode, let message):
                let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? "HTTP \(statusCode)" : "HTTP \(statusCode): \(trimmed)"
            case .invalidResponse:
                return "The models list could not be read."
            case .noUsableModels:
                return "The provider returned no usable enhancement models."
            }
        }
    }

    static let openAIModelsEndpoint = URL(string: "https://api.openai.com/v1/models")!
    // v1beta lists preview models too; the stable v1 list omits them.
    static let geminiModelsEndpoint = URL(string: "https://generativelanguage.googleapis.com/v1beta/models")!
    static let geminiStableModelsEndpoint = URL(string: "https://generativelanguage.googleapis.com/v1/models")!
    static let anthropicModelsEndpoint = URL(string: "https://api.anthropic.com/v1/models")!
    static let groqModelsEndpoint = URL(string: "https://api.groq.com/openai/v1/models")!
    static let cerebrasModelsEndpoint = URL(string: "https://api.cerebras.ai/v1/models")!
    static let mistralModelsEndpoint = URL(string: "https://api.mistral.ai/v1/models")!

    // MARK: - OpenAI

    /// Fetches `GET /v1/models` and returns the chat models usable for enhancement, newest first.
    static func fetchOpenAIModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        let trimmedKey = try validatedKey(apiKey)
        var request = URLRequest(url: openAIModelsEndpoint)
        request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")

        let data = try await perform(request, timeout: timeout)
        let ids = try openAIEnhancementModelIDs(from: data)
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    static func openAIEnhancementModelIDs(from data: Data) throws -> [String] {
        let response: OpenAIModelList
        do {
            response = try JSONDecoder().decode(OpenAIModelList.self, from: data)
        } catch {
            throw CatalogError.invalidResponse
        }

        return
            response.data
            .filter { $0.shutdownDate == nil && isOpenAIEnhancementModel($0.id) }
            .sorted { lhs, rhs in
                if lhs.created != rhs.created { return (lhs.created ?? 0) > (rhs.created ?? 0) }
                return lhs.id.localizedStandardCompare(rhs.id) == .orderedDescending
            }
            .map(\.id)
            .uniqued()
    }

    // The list endpoint has no capability metadata, so filter by ID. Keep `gpt-*` chat
    // models; drop audio, image, realtime, embedding, moderation and other special-purpose
    // models, Responses-only `-pro` models, restricted-access families, and dated snapshots
    // (their undated alias is listed too).
    private static let openAIExcludedTokens = [
        "audio", "realtime", "live", "transcribe", "whisper", "tts", "speech", "translate",
        "image", "dall-e", "sora", "embedding", "moderation", "search", "computer-use",
        "deep-research", "codex", "cyber", "rosalind", "oss", "instruct", "chat-latest",
    ]

    static func isOpenAIEnhancementModel(_ id: String) -> Bool {
        let lowercased = id.lowercased()
        guard lowercased.hasPrefix("gpt-") else { return false }
        if lowercased.hasSuffix("-pro") || lowercased.contains("-pro-") { return false }
        if openAIExcludedTokens.contains(where: { lowercased.contains($0) }) { return false }
        if hasDatedSuffix(lowercased) { return false }
        return true
    }

    // MARK: - Gemini

    /// Fetches the Gemini models lists and returns the text models usable through the
    /// Interactions API, newest first.
    ///
    /// LLMkit sends `-preview` models to `/v1beta/interactions` and everything else to
    /// `/v1/interactions`, and each endpoint only accepts models from its own version's list
    /// (others return 404). So a model is usable when the v1 list has it, or when it is a
    /// `-preview` model on the v1beta list.
    static func fetchGeminiModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        let trimmedKey = try validatedKey(apiKey)
        async let stable = fetchGeminiModels(endpoint: geminiStableModelsEndpoint, apiKey: trimmedKey, timeout: timeout)
        async let beta = fetchGeminiModels(endpoint: geminiModelsEndpoint, apiKey: trimmedKey, timeout: timeout)

        let ids = try await geminiEnhancementModelIDs(stable: stable, beta: beta)
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    private static func fetchGeminiModels(
        endpoint: URL,
        apiKey: String,
        timeout: TimeInterval
    ) async throws -> [GeminiModel] {
        var models: [GeminiModel] = []
        var pageToken: String?

        // models.list is paginated; ten pages of 1000 is far beyond the current catalog.
        for _ in 0..<10 {
            var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
            var queryItems = [URLQueryItem(name: "pageSize", value: "1000")]
            if let pageToken {
                queryItems.append(URLQueryItem(name: "pageToken", value: pageToken))
            }
            components.queryItems = queryItems

            var request = URLRequest(url: components.url!)
            request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

            let page = try decodeGeminiModelList(try await perform(request, timeout: timeout))
            models.append(contentsOf: page.models ?? [])
            guard let next = page.nextPageToken, !next.isEmpty else { break }
            pageToken = next
        }
        return models
    }

    private static func decodeGeminiModelList(_ data: Data) throws -> GeminiModelList {
        do {
            return try JSONDecoder().decode(GeminiModelList.self, from: data)
        } catch {
            throw CatalogError.invalidResponse
        }
    }

    private static func geminiEnhancementModelIDs(stable: [GeminiModel], beta: [GeminiModel]) -> [String] {
        func generateContentIDs(_ models: [GeminiModel]) -> [String] {
            models
                .filter { ($0.supportedGenerationMethods ?? []).contains("generateContent") }
                .map { $0.name.hasPrefix("models/") ? String($0.name.dropFirst("models/".count)) : $0.name }
        }

        let previewIDs = generateContentIDs(beta).filter { $0.lowercased().contains("-preview") }
        return (generateContentIDs(stable) + previewIDs)
            .filter(isGeminiEnhancementModel)
            .sorted { lhs, rhs in
                // Gemini before Gemma, then newest first.
                let lhsIsGemma = lhs.lowercased().hasPrefix("gemma-")
                let rhsIsGemma = rhs.lowercased().hasPrefix("gemma-")
                if lhsIsGemma != rhsIsGemma { return rhsIsGemma }
                return lhs.localizedStandardCompare(rhs) == .orderedDescending
            }
            .uniqued()
    }

    // Both lists also carry image, TTS, music, video, robotics, agent and Live models, many of
    // which accept a text request but aren't text models. Keep only text models.
    private static let geminiExcludedTokens = [
        "embedding", "tts", "image", "imagen", "nano-banana", "veo", "omni", "lyria", "live",
        "audio", "transcribe", "robotics", "computer-use", "deep-research", "antigravity",
        "decision", "aqa", "learnlm", "customtools", "exp",
    ]

    static func isGeminiEnhancementModel(_ id: String) -> Bool {
        let lowercased = id.lowercased()
        guard lowercased.hasPrefix("gemini-") || lowercased.hasPrefix("gemma-") else { return false }
        // Only Gemini 3 and later accept a system instruction on /v1/interactions, and
        // enhancement always sends one.
        if lowercased.hasPrefix("gemini-") {
            guard let major = geminiMajorVersion(lowercased), major >= 3 else { return false }
        }
        // Moving aliases such as `gemini-flash-latest` are v1beta-only without a `-preview`
        // suffix, so LLMkit would send them to v1, which returns 404.
        if lowercased.hasSuffix("-latest") { return false }
        if geminiExcludedTokens.contains(where: { lowercased.contains($0) }) { return false }
        if hasDatedSuffix(lowercased) { return false }
        return true
    }

    // MARK: - Anthropic

    /// Fetches every page of `GET /v1/models` and returns the active Claude models, newest first.
    static func fetchAnthropicModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        let trimmedKey = try validatedKey(apiKey)
        var models: [AnthropicModel] = []
        var afterID: String?

        // Pages hold up to 1000 models; ten pages is far beyond the current catalog.
        for _ in 0..<10 {
            var components = URLComponents(url: anthropicModelsEndpoint, resolvingAgainstBaseURL: false)!
            var queryItems = [URLQueryItem(name: "limit", value: "1000")]
            if let afterID {
                queryItems.append(URLQueryItem(name: "after_id", value: afterID))
            }
            components.queryItems = queryItems

            var request = URLRequest(url: components.url!)
            request.setValue(trimmedKey, forHTTPHeaderField: "x-api-key")
            // The same version LLMkit sends with enhancement requests.
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

            let page = try decode(AnthropicModelList.self, from: try await perform(request, timeout: timeout))
            models.append(contentsOf: page.data)
            guard page.hasMore == true, let next = page.lastID, !next.isEmpty, next != afterID else { break }
            afterID = next
        }

        let ids = anthropicEnhancementModelIDs(models)
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    static func anthropicEnhancementModelIDs(from data: Data) throws -> [String] {
        anthropicEnhancementModelIDs(try decode(AnthropicModelList.self, from: data).data)
    }

    /// Every Claude model is a text chat model on the Messages API, so only the lifecycle
    /// filters: `deprecated` models are closed to new users and `retired` ones fail. The list
    /// is already newest first.
    private static func anthropicEnhancementModelIDs(_ models: [AnthropicModel]) -> [String] {
        models
            .filter { model in
                guard model.type == nil || model.type == "model" else { return false }
                if let lifecycle = model.lifecycle { return lifecycle == "active" }
                return model.deprecatedAt == nil
            }
            .map(\.id)
            .uniqued()
    }

    // MARK: - Groq

    static func fetchGroqModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        try await fetchOpenAICompatibleModelIDs(
            endpoint: groqModelsEndpoint,
            apiKey: apiKey,
            timeout: timeout,
            filter: groqEnhancementModelIDs
        )
    }

    static func groqEnhancementModelIDs(from data: Data) throws -> [String] {
        groqEnhancementModelIDs(try decode(OpenAICompatibleModelList.self, from: data).data)
    }

    // Groq's list has no capability fields, but it marks retired models `active: false`.
    // Besides the shared exclusions, drop the `compound` systems, which run web search
    // and code tools on the request instead of only answering it.
    private static let groqExcludedTokens = ["playai", "orpheus", "compound"]

    private static func groqEnhancementModelIDs(_ models: [OpenAICompatibleModel]) -> [String] {
        models
            .filter { model in
                let lowercased = model.id.lowercased()
                return model.active != false
                    && isOpenAICompatibleChatModel(model.id)
                    && !groqExcludedTokens.contains(where: { lowercased.contains($0) })
            }
            .sortedNewestFirst()
    }

    // MARK: - Cerebras

    static func fetchCerebrasModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        try await fetchOpenAICompatibleModelIDs(
            endpoint: cerebrasModelsEndpoint,
            apiKey: apiKey,
            timeout: timeout,
            filter: cerebrasEnhancementModelIDs
        )
    }

    static func cerebrasEnhancementModelIDs(from data: Data) throws -> [String] {
        cerebrasEnhancementModelIDs(try decode(OpenAICompatibleModelList.self, from: data).data)
    }

    /// Cerebras only serves text chat models and its list has no capability fields, so
    /// only the shared exclusions apply, in case it adds other kinds of model.
    private static func cerebrasEnhancementModelIDs(_ models: [OpenAICompatibleModel]) -> [String] {
        models
            .filter { $0.active != false && isOpenAICompatibleChatModel($0.id) }
            .sortedNewestFirst()
    }

    // MARK: - Mistral

    static func fetchMistralModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        let trimmedKey = try validatedKey(apiKey)
        var request = URLRequest(url: mistralModelsEndpoint)
        request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")

        let ids = try mistralEnhancementModelIDs(from: try await perform(request, timeout: timeout))
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    /// Keeps the chat models from Mistral's capability flags, and lists each model once.
    static func mistralEnhancementModelIDs(from data: Data) throws -> [String] {
        let models = try decode(MistralModelList.self, from: data).data.filter(isMistralEnhancementModel)

        // The list has a card for every alias of a model (`mistral-small-latest` and
        // `mistral-small-2506`), all with the same `name`. Keep one ID per model: the moving
        // `-latest` alias when there is one, as the built-in list does, or else the model name.
        var groups: [String: [MistralModel]] = [:]
        var order: [String] = []
        for model in models {
            let key = model.name ?? model.id
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(model)
        }

        return order.compactMap { key -> (id: String, created: Int)? in
            guard let cards = groups[key] else { return nil }
            let ids = cards.map(\.id)
            let id = ids.first(where: { $0.hasSuffix("-latest") })
                ?? ids.first(where: { $0 == key })
                ?? ids.sorted().first
            guard let id else { return nil }
            return (id, cards.compactMap(\.created).max() ?? 0)
        }
        .sorted { lhs, rhs in
            if lhs.created != rhs.created { return lhs.created > rhs.created }
            return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
        }
        .map(\.id)
        .uniqued()
    }

    // Models that always reason return `message.content` as a list of thinking and text
    // chunks, whatever `reasoning_effort` says, and LLMkit only reads a plain string. The
    // `reasoning` capability is also set on hybrid models that answer in plain text by
    // default, so these are matched by ID. Mistral Large 4 "always reasons" and defaults
    // to `high`; `mistral-large-latest` points to it. See
    // https://docs.mistral.ai/inference/conversations/reasoning
    private static let mistralAlwaysReasoningTokens = [
        "zai-glm", "magistral", "mistral-large-4", "mistral-large-latest",
    ]

    private static func isMistralEnhancementModel(_ model: MistralModel) -> Bool {
        guard let capabilities = model.capabilities, capabilities.completionChat == true else { return false }
        let ids = [model.id, model.name ?? ""].map { $0.lowercased() }
        if ids.contains(where: { id in mistralAlwaysReasoningTokens.contains(where: { id.contains($0) }) }) {
            return false
        }
        // OCR, moderation, classifier, transcription and speech models can also be
        // flagged for chat; none is a plain text model.
        let specialPurpose = [
            capabilities.ocr, capabilities.moderation, capabilities.classification,
            capabilities.audio, capabilities.audioTranscription, capabilities.audioTranscriptionRealtime,
            capabilities.audioSpeech,
        ]
        if specialPurpose.contains(true) { return false }
        // `deprecation` is the scheduled retirement date, set once a model is deprecated.
        if model.deprecation != nil { return false }
        if model.archived == true { return false }
        return true
    }

    // MARK: - Custom OpenAI-compatible providers

    /// The `/models` URL next to a configured chat completions URL, or nil when the URL
    /// doesn't end in `/chat/completions` and so has no known models endpoint.
    static func customModelsEndpoint(forChatCompletionsURL urlString: String) -> URL? {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var components = URLComponents(string: trimmed),
            let scheme = components.scheme?.lowercased(), scheme == "http" || scheme == "https",
            components.host?.isEmpty == false
        else {
            return nil
        }

        var path = components.path
        while path.hasSuffix("/") { path.removeLast() }
        let suffix = "/chat/completions"
        guard path.lowercased().hasSuffix(suffix) else { return nil }
        path.removeLast(suffix.count)
        components.path = path + "/models"
        return components.url
    }

    /// Fetches the models a custom OpenAI-compatible endpoint lists. Many local servers
    /// support this, but not all; callers treat a failure as "no list" and keep the
    /// user's typed model.
    static func fetchCustomModelIDs(
        chatCompletionsURL: String,
        apiKey: String,
        timeout: TimeInterval = 10
    ) async throws -> [String] {
        guard let endpoint = customModelsEndpoint(forChatCompletionsURL: chatCompletionsURL) else {
            throw CatalogError.invalidResponse
        }
        return try await fetchOpenAICompatibleModelIDs(
            endpoint: endpoint,
            apiKey: apiKey,
            timeout: timeout,
            filter: customEnhancementModelIDs
        )
    }

    static func customEnhancementModelIDs(from data: Data) throws -> [String] {
        customEnhancementModelIDs(try decode(OpenAICompatibleModelList.self, from: data).data)
    }

    private static func customEnhancementModelIDs(_ models: [OpenAICompatibleModel]) -> [String] {
        models
            .filter { $0.active != false && isOpenAICompatibleChatModel($0.id) }
            .map(\.id)
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
            .uniqued()
    }

    // MARK: - Together AI

    // `serverless=true` leaves out the models that only run on a dedicated endpoint, which
    // return 400 for a serverless key.
    static let togetherAIModelsEndpoint = URL(string: "https://api.together.xyz/v1/models?serverless=true")!

    static func fetchTogetherAIModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        var request = URLRequest(url: togetherAIModelsEndpoint)
        request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")
        let ids = try togetherAIEnhancementModelIDs(from: try await perform(request, timeout: timeout))
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    /// Together returns a bare array with a `type` per model (`chat`, `image`, `audio`,
    /// `embedding`, `rerank`, …). Keeps the chat models.
    static func togetherAIEnhancementModelIDs(from data: Data) throws -> [String] {
        try decode([TogetherAIModel].self, from: data)
            .filter { model in
                model.type == "chat"
                    && isOpenAICompatibleChatModel(model.id)
                    && !isUnusableHostedModel(model.id, for: .togetherAI)
            }
            .sorted { lhs, rhs in
                if lhs.created != rhs.created { return (lhs.created ?? 0) > (rhs.created ?? 0) }
                return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
            }
            .map(\.id)
            .uniqued()
    }

    // MARK: - GMI Cloud

    static let gmiCloudModelsEndpoint = URL(string: "https://api.gmi-serving.com/v1/models")!

    static func fetchGMICloudModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        try await fetchOpenAICompatibleModelIDs(
            endpoint: gmiCloudModelsEndpoint,
            apiKey: apiKey,
            timeout: timeout,
            filter: gmiCloudEnhancementModelIDs
        )
    }

    static func gmiCloudEnhancementModelIDs(from data: Data) throws -> [String] {
        gmiCloudEnhancementModelIDs(try decode(OpenAICompatibleModelList.self, from: data).data)
    }

    /// GMI's list has no capability fields and only serves text chat models today, so the
    /// shared exclusions apply, plus the models that failed VoiceInk's request.
    private static func gmiCloudEnhancementModelIDs(_ models: [OpenAICompatibleModel]) -> [String] {
        models
            .filter { isOpenAICompatibleChatModel($0.id) && !isUnusableHostedModel($0.id, for: .gmiCloud) }
            .sortedNewestFirst()
    }

    // MARK: - Baseten

    static let basetenModelsEndpoint = URL(string: "https://inference.baseten.co/v1/models")!

    static func fetchBasetenModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        var request = URLRequest(url: basetenModelsEndpoint)
        request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")
        let ids = try basetenEnhancementModelIDs(from: try await perform(request, timeout: timeout))
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    /// Baseten's Model APIs list carries input and output modalities. Keeps the models that
    /// take and return text.
    static func basetenEnhancementModelIDs(from data: Data) throws -> [String] {
        try decode(BasetenModelList.self, from: data).data
            .filter { model in
                (model.inputModalities ?? ["text"]).contains("text")
                    && (model.outputModalities ?? ["text"]) == ["text"]
                    && isOpenAICompatibleChatModel(model.id)
                    && !isUnusableHostedModel(model.id, for: .baseten)
            }
            .sorted { lhs, rhs in
                if lhs.created != rhs.created { return (lhs.created ?? 0) > (rhs.created ?? 0) }
                return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
            }
            .map(\.id)
            .uniqued()
    }

    // MARK: - Fireworks AI

    static let fireworksModelsEndpoint = URL(string: "https://api.fireworks.ai/inference/v1/models")!

    static func fetchFireworksModelIDs(apiKey: String, timeout: TimeInterval = 15) async throws -> [String] {
        var request = URLRequest(url: fireworksModelsEndpoint)
        request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")
        let ids = try fireworksEnhancementModelIDs(from: try await perform(request, timeout: timeout))
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    /// Fireworks marks chat models with `supports_chat` and embedding and reranker models
    /// with `kind: EMBEDDING_MODEL`.
    static func fireworksEnhancementModelIDs(from data: Data) throws -> [String] {
        try decode(FireworksModelList.self, from: data).data
            .filter { model in
                model.supportsChat == true
                    && model.kind != "EMBEDDING_MODEL"
                    && isOpenAICompatibleChatModel(model.id)
                    && !isUnusableHostedModel(model.id, for: .fireworks)
            }
            .sorted { lhs, rhs in
                if lhs.created != rhs.created { return (lhs.created ?? 0) > (rhs.created ?? 0) }
                return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
            }
            .map(\.id)
            .uniqued()
    }

    // MARK: - Vercel AI Gateway

    // Public: the list answers without a key, so it can't verify one.
    static let vercelAIGatewayModelsEndpoint = URL(string: "https://ai-gateway.vercel.sh/v1/models")!

    /// A Vercel AI Gateway model VoiceInk can use, with the reasoning efforts its catalog
    /// entry lists.
    struct VercelAIGatewayModel: Equatable, Sendable {
        let id: String
        let reasoningEfforts: [String]
    }

    static func fetchVercelAIGatewayModels(apiKey: String, timeout: TimeInterval = 15) async throws
        -> [VercelAIGatewayModel]
    {
        var request = URLRequest(url: vercelAIGatewayModelsEndpoint)
        request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")
        let models = try vercelAIGatewayEnhancementModels(from: try await perform(request, timeout: timeout))
        guard !models.isEmpty else { throw CatalogError.noUsableModels }
        return models
    }

    /// Keeps the `language` models that return only text. The list also has image, video,
    /// embedding, evaluation, speech, transcription, realtime and reranking models, and
    /// `-batch` aliases that run through the batch API.
    static func vercelAIGatewayEnhancementModels(
        from data: Data,
        now: Date = Date()
    ) throws -> [VercelAIGatewayModel] {
        try decode(VercelAIGatewayModelList.self, from: data).data
            .filter { model in
                model.type == "language"
                    && (model.modalities?.output ?? ["text"]) == ["text"]
                    && !model.isRetired(asOf: now)
                    && !model.id.lowercased().hasSuffix("-batch")
                    && !GeminiModelMigration.isUnsupported(model.id)
                    && isOpenAICompatibleChatModel(model.id)
                    && !isUnusableHostedModel(model.id, for: .vercelAIGateway)
            }
            .sorted { lhs, rhs in
                let lhsOwner = lhs.id.split(separator: "/").first.map(String.init) ?? lhs.id
                let rhsOwner = rhs.id.split(separator: "/").first.map(String.init) ?? rhs.id
                if lhsOwner != rhsOwner { return lhsOwner.localizedStandardCompare(rhsOwner) == .orderedAscending }
                if lhs.released != rhs.released { return (lhs.released ?? 0) > (rhs.released ?? 0) }
                return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
            }
            .reduce(into: [VercelAIGatewayModel]()) { models, model in
                guard !models.contains(where: { $0.id == model.id }) else { return }
                let efforts = (model.reasoningOptions ?? [])
                    .filter { $0.type == "effort" }
                    .flatMap { $0.values ?? [] }
                models.append(VercelAIGatewayModel(id: model.id, reasoningEfforts: efforts))
            }
    }

    // MARK: - Key verification for Together AI, GMI Cloud, Baseten, Fireworks AI, Vercel AI Gateway

    /// An authenticated GET that needs no model ID. Vercel's model list is public, so its key
    /// is checked against `/v1/credits` instead.
    static func keyVerificationEndpoint(for provider: AIProvider) -> URL? {
        switch provider {
        case .togetherAI: return URL(string: "https://api.together.xyz/v1/models")
        case .gmiCloud: return gmiCloudModelsEndpoint
        case .baseten: return basetenModelsEndpoint
        case .fireworks: return fireworksModelsEndpoint
        case .vercelAIGateway: return URL(string: "https://ai-gateway.vercel.sh/v1/credits")
        default: return nil
        }
    }

    static func verifyAPIKey(_ apiKey: String, for provider: AIProvider, timeout: TimeInterval = 10) async -> (
        isValid: Bool, errorMessage: String?
    ) {
        guard let endpoint = keyVerificationEndpoint(for: provider) else {
            return (false, "Invalid or missing base URL configuration")
        }
        do {
            var request = URLRequest(url: endpoint)
            request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")
            _ = try await perform(request, timeout: timeout)
            return (true, nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }

    // MARK: - OpenAI-compatible lists

    // OpenAI-style lists carry no capability fields, so special-purpose models are
    // recognised by name: embeddings, speech-to-text, text-to-speech, moderation and
    // safety classifiers, rerankers and image generators.
    private static let openAICompatibleExcludedTokens = [
        "embed", "whisper", "transcribe", "tts", "speech", "audio", "moderation", "guard",
        "rerank", "dall-e", "image-gen", "gpt-image", "stable-diffusion", "flux",
    ]

    static func isOpenAICompatibleChatModel(_ id: String) -> Bool {
        let lowercased = id.lowercased()
        return !lowercased.isEmpty && !openAICompatibleExcludedTokens.contains(where: { lowercased.contains($0) })
    }

    private static func fetchOpenAICompatibleModelIDs(
        endpoint: URL,
        apiKey: String,
        timeout: TimeInterval,
        filter: ([OpenAICompatibleModel]) -> [String]
    ) async throws -> [String] {
        var request = URLRequest(url: endpoint)
        request.setValue("Bearer \(try validatedKey(apiKey))", forHTTPHeaderField: "Authorization")

        let ids = filter(try decode(OpenAICompatibleModelList.self, from: try await perform(request, timeout: timeout)).data)
        guard !ids.isEmpty else { throw CatalogError.noUsableModels }
        return ids
    }

    // MARK: - Helpers

    /// The major version in a `gemini-<major>[.<minor>]-…` ID, or nil for unversioned IDs.
    private static func geminiMajorVersion(_ id: String) -> Int? {
        guard let range = id.range(of: #"^gemini-\d+"#, options: .regularExpression) else { return nil }
        return Int(id[range].dropFirst("gemini-".count))
    }

    /// Matches pinned snapshot suffixes: `-2025-04-14`, `-0613`, `-001`, `-09-2025`.
    private static func hasDatedSuffix(_ id: String) -> Bool {
        let patterns = [#"-\d{4}-\d{2}-\d{2}$"#, #"-\d{3,4}$"#]
        return patterns.contains { id.range(of: $0, options: .regularExpression) != nil }
    }

    private static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw CatalogError.invalidResponse
        }
    }

    private static func validatedKey(_ apiKey: String) throws -> String {
        let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw CatalogError.missingAPIKey }
        return trimmed
    }

    private static func perform(_ request: URLRequest, timeout: TimeInterval) async throws -> Data {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw CatalogError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw CatalogError.httpError(
                statusCode: http.statusCode,
                message: apiErrorMessage(from: data) ?? String(data: data, encoding: .utf8) ?? ""
            )
        }
        return data
    }

    /// Most providers wrap failures as `{"error": {"message": "..."}}`; Mistral and some
    /// OpenAI-compatible servers use a top-level `message` or `detail` string.
    private static func apiErrorMessage(from data: Data) -> String? {
        struct ErrorBody: Decodable {
            struct Detail: Decodable { let message: String? }
            let error: Detail?
            let message: String?
            let detail: String?
        }
        guard let body = try? JSONDecoder().decode(ErrorBody.self, from: data) else { return nil }
        return body.error?.message ?? body.message ?? body.detail
    }
}

// MARK: - Cache

/// Shared cache of the fetched model lists, so every `AIService` instance sees the same list
/// and concurrent refreshes for one provider make a single request.
final class LiveModelListStore: @unchecked Sendable {
    static let shared = LiveModelListStore()
    static let didChangeNotification = Notification.Name("LiveModelListStoreDidChange")

    private let lock = NSLock()
    private let defaults = UserDefaults.standard
    private var lists: [String: [String]] = [:]
    private var inFlight: [String: Task<[String], Error>] = [:]

    private init() {}

    private static func modelsKey(_ providerKey: String) -> String {
        "\(providerKey)LiveEnhancementModels"
    }

    private static func fetchedAtKey(_ providerKey: String) -> String {
        "\(providerKey)LiveEnhancementModelsFetchedAt"
    }

    func models(forProviderKey providerKey: String) -> [String]? {
        lock.withLock {
            if let cached = lists[providerKey] { return cached }
            guard let saved = defaults.stringArray(forKey: Self.modelsKey(providerKey)), !saved.isEmpty else {
                return nil
            }
            lists[providerKey] = saved
            return saved
        }
    }

    func fetchedAt(forProviderKey providerKey: String) -> Date? {
        defaults.object(forKey: Self.fetchedAtKey(providerKey)) as? Date
    }

    /// Saves a fetched list. Returns whether the list changed.
    @discardableResult
    func save(_ models: [String], forProviderKey providerKey: String) -> Bool {
        let changed = models != self.models(forProviderKey: providerKey)
        lock.withLock {
            lists[providerKey] = models
            defaults.set(models, forKey: Self.modelsKey(providerKey))
            defaults.set(Date(), forKey: Self.fetchedAtKey(providerKey))
        }
        if changed {
            NotificationCenter.default.post(name: Self.didChangeNotification, object: nil)
        }
        return changed
    }

    /// Drops a cached list, for example when its API key is removed. Returns whether one existed.
    @discardableResult
    func clear(forProviderKey providerKey: String) -> Bool {
        let existed = models(forProviderKey: providerKey) != nil
        lock.withLock {
            lists.removeValue(forKey: providerKey)
            defaults.removeObject(forKey: Self.modelsKey(providerKey))
            defaults.removeObject(forKey: Self.fetchedAtKey(providerKey))
        }
        if existed {
            NotificationCenter.default.post(name: Self.didChangeNotification, object: nil)
        }
        return existed
    }

    /// Runs `operation`, or joins the request already in flight for this provider and key.
    func fetch(
        forProviderKey providerKey: String,
        apiKey: String,
        operation: @escaping @Sendable () async throws -> [String]
    ) async throws -> [String] {
        let requestKey = "\(providerKey)\n\(apiKey)"
        let (task, isOwner): (Task<[String], Error>, Bool) = lock.withLock {
            if let existing = inFlight[requestKey] { return (existing, false) }
            let task = Task { try await operation() }
            inFlight[requestKey] = task
            return (task, true)
        }
        defer {
            if isOwner {
                lock.withLock { inFlight[requestKey] = nil }
            }
        }
        return try await task.value
    }
}

// MARK: - Response Models

private struct OpenAIModelList: Decodable {
    let data: [OpenAIModel]
}

private struct OpenAIModel: Decodable {
    let id: String
    let created: Int?
    let shutdownDate: String?

    enum CodingKeys: String, CodingKey {
        case id
        case created
        case shutdownDate = "shutdown_date"
    }
}

private struct GeminiModelList: Decodable {
    let models: [GeminiModel]?
    let nextPageToken: String?
}

private struct GeminiModel: Decodable {
    let name: String
    let supportedGenerationMethods: [String]?
}

private struct AnthropicModelList: Decodable {
    let data: [AnthropicModel]
    let hasMore: Bool?
    let lastID: String?

    enum CodingKeys: String, CodingKey {
        case data
        case hasMore = "has_more"
        case lastID = "last_id"
    }
}

private struct AnthropicModel: Decodable {
    let id: String
    let type: String?
    let lifecycle: String?
    let deprecatedAt: String?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case lifecycle
        case deprecatedAt = "deprecated_at"
    }
}

/// The OpenAI-style list that Groq, Cerebras and custom servers return. Groq adds `active`.
private struct OpenAICompatibleModelList: Decodable {
    let data: [OpenAICompatibleModel]
}

private struct OpenAICompatibleModel: Decodable {
    let id: String
    let created: Int?
    let active: Bool?
}

private struct MistralModelList: Decodable {
    let data: [MistralModel]
}

private struct MistralModel: Decodable {
    let id: String
    let name: String?
    let created: Int?
    let capabilities: MistralModelCapabilities?
    let deprecation: String?
    let archived: Bool?
}

private struct MistralModelCapabilities: Decodable {
    let completionChat: Bool?
    let ocr: Bool?
    let moderation: Bool?
    let classification: Bool?
    let audio: Bool?
    let audioTranscription: Bool?
    let audioTranscriptionRealtime: Bool?
    let audioSpeech: Bool?

    enum CodingKeys: String, CodingKey {
        case completionChat = "completion_chat"
        case ocr
        case moderation
        case classification
        case audio
        case audioTranscription = "audio_transcription"
        case audioTranscriptionRealtime = "audio_transcription_realtime"
        case audioSpeech = "audio_speech"
    }
}

private struct TogetherAIModel: Decodable {
    let id: String
    let type: String?
    let created: Int?
}

private struct BasetenModelList: Decodable {
    let data: [BasetenModel]
}

private struct BasetenModel: Decodable {
    let id: String
    let created: Int?
    let inputModalities: [String]?
    let outputModalities: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case created
        case inputModalities = "input_modalities"
        case outputModalities = "output_modalities"
    }
}

private struct FireworksModelList: Decodable {
    let data: [FireworksModel]
}

private struct FireworksModel: Decodable {
    let id: String
    let created: Int?
    let kind: String?
    let supportsChat: Bool?

    enum CodingKeys: String, CodingKey {
        case id
        case created
        case kind
        case supportsChat = "supports_chat"
    }
}

private struct VercelAIGatewayModelList: Decodable {
    let data: [VercelAIGatewayCatalogModel]
}

private struct VercelAIGatewayCatalogModel: Decodable {
    struct Modalities: Decodable {
        let output: [String]?
    }

    struct ReasoningOption: Decodable {
        let type: String
        let values: [String]?
    }

    let id: String
    let type: String?
    let released: Int?
    let modalities: Modalities?
    let reasoningOptions: [ReasoningOption]?
    /// When the gateway stops serving the model, in milliseconds since 1970. Set ahead of time,
    /// so a model with a future date still works.
    let deprecatedAt: Double?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case released
        case modalities
        case reasoningOptions = "reasoning_options"
        case deprecatedAt = "deprecated_at"
    }

    func isRetired(asOf date: Date) -> Bool {
        guard let deprecatedAt else { return false }
        return Date(timeIntervalSince1970: deprecatedAt / 1000) <= date
    }
}

extension Array where Element == String {
    fileprivate func uniqued() -> [String] {
        var seen = Set<String>()
        return filter { seen.insert($0).inserted }
    }
}

extension Array where Element == OpenAICompatibleModel {
    /// IDs sorted newest first by `created`, then by name.
    fileprivate func sortedNewestFirst() -> [String] {
        sorted { lhs, rhs in
            if lhs.created != rhs.created { return (lhs.created ?? 0) > (rhs.created ?? 0) }
            return lhs.id.localizedStandardCompare(rhs.id) == .orderedAscending
        }
        .map(\.id)
        .uniqued()
    }
}
