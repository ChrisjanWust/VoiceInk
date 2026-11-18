import Foundation

/// Fetches the live model lists for hosted providers whose static presets go stale,
/// and keeps only the text models that VoiceInk can use for enhancement.
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
        // Gemini 1.x and 2.x reject a system instruction on /v1/interactions ("Developer
        // instruction is not enabled for api version v1"), and enhancement always sends one.
        if lowercased.range(of: #"^gemini-[12]\."#, options: .regularExpression) != nil { return false }
        // Moving aliases such as `gemini-flash-latest` are v1beta-only without a `-preview`
        // suffix, so LLMkit would send them to v1, which returns 404.
        if lowercased.hasSuffix("-latest") { return false }
        if geminiExcludedTokens.contains(where: { lowercased.contains($0) }) { return false }
        if hasDatedSuffix(lowercased) { return false }
        return true
    }

    // MARK: - Helpers

    /// Matches pinned snapshot suffixes: `-2025-04-14`, `-0613`, `-001`, `-09-2025`.
    private static func hasDatedSuffix(_ id: String) -> Bool {
        let patterns = [#"-\d{4}-\d{2}-\d{2}$"#, #"-\d{3,4}$"#]
        return patterns.contains { id.range(of: $0, options: .regularExpression) != nil }
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

    /// Both providers wrap failures as `{"error": {"message": "..."}}`.
    private static func apiErrorMessage(from data: Data) -> String? {
        struct ErrorBody: Decodable {
            struct Detail: Decodable { let message: String? }
            let error: Detail?
        }
        return (try? JSONDecoder().decode(ErrorBody.self, from: data))?.error?.message
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

extension Array where Element == String {
    fileprivate func uniqued() -> [String] {
        var seen = Set<String>()
        return filter { seen.insert($0).inserted }
    }
}
