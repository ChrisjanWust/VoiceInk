import Foundation

/// One Amazon Bedrock text model that VoiceInk can use for enhancement, from the account's
/// live model list.
struct BedrockModelInfo: Codable, Sendable, Equatable {
    /// The ID Converse is called with: the base model ID for on-demand models, or a
    /// system-defined inference profile ID for models that only run through a profile.
    let invocationID: String
    /// The foundation model ID, such as `anthropic.claude-sonnet-5-5`.
    let modelID: String
    let providerName: String
    let modelName: String
    let releasedAt: Date?
    /// `converse.additionalRequestFieldsSchema`: a JSON string describing the extra request
    /// fields the model accepts, including its reasoning control. `nil` when not listed.
    let additionalRequestFieldsSchema: String?
    /// `converse.reasoningSupported.embedded`: whether the model writes its reasoning into the
    /// text itself rather than a separate content block. `nil` when not listed.
    let embeddedReasoning: Bool?
    /// `converse.maxTokensDefault` and `maxTokensMaximum`: the default and largest output limits.
    let maxTokensDefault: Int?
    let maxTokensMaximum: Int?

    /// For display, such as "Anthropic · Claude Sonnet 5.5".
    var displayName: String {
        "\(providerName) · \(modelName)"
    }
}

/// Fetches the account's Bedrock models with a Bedrock API key and keeps the ones that work
/// with VoiceInk's enhancement request.
///
/// Both list calls accept the API key as a bearer token:
/// - `GET https://bedrock.{region}.amazonaws.com/foundation-models?byOutputModality=TEXT`
/// - `GET https://bedrock.{region}.amazonaws.com/inference-profiles?typeEquals=SYSTEM_DEFINED`
enum BedrockModelCatalog {
    static func controlPlaneEndpoint(for region: String) -> String {
        "https://bedrock.\(BedrockRegion.normalized(region)).amazonaws.com"
    }

    /// Fetches the usable models, saves their details for the reasoning picker, and returns
    /// their invocation IDs in display order.
    static func fetchModelIDs(apiKey: String, region: String, timeout: TimeInterval = 15) async throws -> [String] {
        let models = try await fetchModels(apiKey: apiKey, region: region, timeout: timeout)
        BedrockModelDetailsStore.shared.save(models, region: region)
        return models.map(\.invocationID)
    }

    static func fetchModels(apiKey: String, region: String, timeout: TimeInterval = 15) async throws -> [BedrockModelInfo] {
        let key = BedrockConverseClient.normalizedAPIKey(apiKey)
        guard !key.isEmpty else { throw ProviderModelCatalog.CatalogError.missingAPIKey }
        let normalizedRegion = BedrockRegion.normalized(region)
        guard BedrockRegion.isValid(normalizedRegion) else { throw BedrockError.invalidRegion(region) }

        async let foundationModels: [FoundationModelSummary] = fetchPages(
            path: "/foundation-models",
            query: [URLQueryItem(name: "byOutputModality", value: "TEXT")],
            apiKey: key,
            region: normalizedRegion,
            timeout: timeout
        ) { (page: FoundationModelList) in (page.modelSummaries ?? [], page.nextToken) }

        async let profiles: [InferenceProfileSummary] = fetchPages(
            path: "/inference-profiles",
            query: [
                URLQueryItem(name: "maxResults", value: "1000"),
                URLQueryItem(name: "typeEquals", value: "SYSTEM_DEFINED"),
            ],
            apiKey: key,
            region: normalizedRegion,
            timeout: timeout
        ) { (page: InferenceProfileList) in (page.inferenceProfileSummaries ?? [], page.nextToken) }

        let models = usableModels(foundationModels: try await foundationModels, profiles: try await profiles)
        guard !models.isEmpty else { throw ProviderModelCatalog.CatalogError.noUsableModels }
        return models
    }

    // MARK: - Filtering

    /// Keeps active text models that Converse can call with a system prompt, picks the ID to
    /// call each one with, and orders them for the picker.
    static func usableModels(
        foundationModels: [FoundationModelSummary],
        profiles: [InferenceProfileSummary]
    ) -> [BedrockModelInfo] {
        let activeProfiles = profiles.filter { ($0.status ?? "ACTIVE") == "ACTIVE" }

        let models = foundationModels.compactMap { model -> BedrockModelInfo? in
            guard model.modelLifecycle?.status == "ACTIVE",
                (model.inputModalities ?? []).contains("TEXT"),
                (model.outputModalities ?? []).contains("TEXT"),
                // Enhancement always sends a system prompt.
                let converse = model.converse, converse.systemRoleSupported == true,
                !excludedModelIDs.contains(model.modelId),
                let invocationID = invocationID(for: model, profiles: activeProfiles)
            else {
                return nil
            }
            return BedrockModelInfo(
                invocationID: invocationID,
                modelID: model.modelId,
                providerName: model.providerName ?? "",
                modelName: model.modelName ?? model.modelId,
                releasedAt: model.modelLifecycle?.startOfLifeTime.flatMap(parseDate),
                additionalRequestFieldsSchema: converse.additionalRequestFieldsSchema,
                embeddedReasoning: converse.reasoningSupported?.embedded,
                maxTokensDefault: converse.maxTokensDefault,
                maxTokensMaximum: converse.maxTokensMaximum
            )
        }
        return ordered(uniqued(models))
    }

    /// Models the filter above lets through that failed VoiceInk's Default enhancement request
    /// in live tests on 2026-11-18 (us-east-2, 3 to 7 runs each). Nothing in the catalog marks
    /// them, so they're listed here. They still work as custom model IDs.
    static let excludedModelIDs: Set<String> = [
        // Start with a "Here's the cleaned transcript:" style preamble, which would be pasted.
        "amazon.nova-pro-v1:0",
        "minimax.minimax-m2",
        "minimax.minimax-m2.1",
        "mistral.devstral-2-123b",
        "mistral.magistral-small-2509",
        "mistral.ministral-3-3b-instruct",
        "mistral.ministral-3-8b-instruct",
        "mistral.ministral-3-14b-instruct",
        "mistral.mistral-large-3-675b-instruct",
        "mistral.voxtral-mini-3b-2507",
        "nvidia.nemotron-nano-12b-v2",
        "qwen.qwen3-32b-v1:0",
        // Writes its reasoning into the answer text.
        "nvidia.nemotron-nano-9b-v2",
        // Ignores the transcript and answers from the prompt's example instead.
        "moonshotai.kimi-k3",
        // Rejects any request with a system prompt: "Conversation roles must alternate".
        "writer.palmyra-vision-7b",
        // Gated: "not available for this account", with access through AWS Sales.
        "zai.glm-5.3",
    ]

    /// On-demand models are called by their base ID. Models that only run through an
    /// inference profile use a system-defined profile in this Region that routes to them,
    /// preferring `global.` (works from most commercial Regions) over geographic profiles.
    static func invocationID(for model: FoundationModelSummary, profiles: [InferenceProfileSummary]) -> String? {
        let inferenceTypes = model.inferenceTypesSupported ?? []
        if inferenceTypes.contains("ON_DEMAND") {
            return model.modelId
        }
        guard inferenceTypes.contains("INFERENCE_PROFILE") else { return nil }

        let modelSuffix = "/\(model.modelId)"
        let candidates = profiles
            .filter { profile in
                (profile.models ?? []).contains { ($0.modelArn ?? "").hasSuffix(modelSuffix) }
            }
            .map(\.inferenceProfileId)
            .sorted()
        return candidates.first { $0.hasPrefix("global.") } ?? candidates.first
    }

    /// The default model first, then by provider, newest first within each provider.
    static func ordered(_ models: [BedrockModelInfo]) -> [BedrockModelInfo] {
        let defaultModel = AIProvider.bedrock.defaultModel
        return models.sorted { lhs, rhs in
            let lhsIsDefault = lhs.invocationID == defaultModel
            let rhsIsDefault = rhs.invocationID == defaultModel
            if lhsIsDefault != rhsIsDefault { return lhsIsDefault }

            let providerOrder = lhs.providerName.localizedStandardCompare(rhs.providerName)
            if providerOrder != .orderedSame { return providerOrder == .orderedAscending }

            switch (lhs.releasedAt, rhs.releasedAt) {
            case let (lhsDate?, rhsDate?) where lhsDate != rhsDate:
                return lhsDate > rhsDate
            case (.some, nil):
                return true
            case (nil, .some):
                return false
            default:
                return lhs.modelName.localizedStandardCompare(rhs.modelName) == .orderedDescending
            }
        }
    }

    private static func uniqued(_ models: [BedrockModelInfo]) -> [BedrockModelInfo] {
        var seen = Set<String>()
        return models.filter { seen.insert($0.invocationID).inserted }
    }

    private static func parseDate(_ value: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: value) { return date }
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: value)
    }

    // MARK: - Requests

    private static func fetchPages<Page: Decodable, Item>(
        path: String,
        query: [URLQueryItem],
        apiKey: String,
        region: String,
        timeout: TimeInterval,
        items: (Page) -> ([Item], String?)
    ) async throws -> [Item] {
        var results: [Item] = []
        var nextToken: String?

        // Both lists fit on one page today; ten pages is far beyond the current catalog.
        for _ in 0..<10 {
            guard var components = URLComponents(string: controlPlaneEndpoint(for: region) + path) else {
                throw BedrockError.invalidRegion(region)
            }
            var queryItems = query
            if let nextToken {
                queryItems.append(URLQueryItem(name: "nextToken", value: nextToken))
            }
            components.queryItems = queryItems
            guard let url = components.url else { throw BedrockError.invalidRegion(region) }

            var request = URLRequest(url: url)
            request.setValue("application/json", forHTTPHeaderField: "Accept")
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

            let data = try await perform(request, timeout: timeout, region: region)
            let page: Page
            do {
                page = try JSONDecoder().decode(Page.self, from: data)
            } catch {
                throw ProviderModelCatalog.CatalogError.invalidResponse
            }
            let (pageItems, token) = items(page)
            results.append(contentsOf: pageItems)
            guard let token, !token.isEmpty else { break }
            nextToken = token
        }
        return results
    }

    private static func perform(_ request: URLRequest, timeout: TimeInterval, region: String) async throws -> Data {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as NSError where error.domain == NSURLErrorDomain {
            switch error.code {
            case NSURLErrorCannotFindHost, NSURLErrorDNSLookupFailed:
                throw BedrockError.unreachableRegion(region: region, detail: error.localizedDescription)
            default:
                throw error
            }
        }

        guard let http = response as? HTTPURLResponse else {
            throw ProviderModelCatalog.CatalogError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let parsed = BedrockErrorDescriber.parse(
                data: data,
                errorTypeHeader: http.value(forHTTPHeaderField: "x-amzn-ErrorType")
            )
            throw ListError(statusCode: http.statusCode, errorType: parsed.errorType, message: parsed.message, region: region)
        }
        return data
    }

    /// A failed list call. The built-in list stays in use, and model IDs can still be typed in.
    struct ListError: LocalizedError, Equatable {
        let statusCode: Int
        let errorType: String?
        let message: String
        let region: String

        var errorDescription: String? {
            let awsSaid = message.isEmpty ? "" : " AWS said: \(message)"
            if BedrockErrorDescriber.isAuthError(statusCode: statusCode, errorType: errorType, message: message) {
                return BedrockErrorDescriber.describe(
                    statusCode: statusCode,
                    errorType: errorType,
                    message: message,
                    modelID: "",
                    region: region
                )
            }
            if statusCode == 403 || errorType == "AccessDeniedException" {
                return
                    "This API key isn't allowed to list Amazon Bedrock models in \(region). Its IAM permissions need bedrock:ListFoundationModels and bedrock:ListInferenceProfiles. You can still enter a model ID."
                    + awsSaid
            }
            if statusCode == 429 || errorType == "ThrottlingException" {
                return "Amazon Bedrock is throttling requests. Try again shortly." + awsSaid
            }
            return "Amazon Bedrock couldn't list models (HTTP \(statusCode))." + awsSaid
        }
    }

    // MARK: - Response models

    struct FoundationModelList: Decodable {
        let modelSummaries: [FoundationModelSummary]?
        let nextToken: String?
    }

    struct FoundationModelSummary: Decodable {
        struct Lifecycle: Decodable {
            let status: String?
            let startOfLifeTime: String?
        }

        struct Converse: Decodable {
            struct ReasoningSupport: Decodable {
                let embedded: Bool?
            }

            let systemRoleSupported: Bool?
            let additionalRequestFieldsSchema: String?
            let reasoningSupported: ReasoningSupport?
            let maxTokensDefault: Int?
            let maxTokensMaximum: Int?
        }

        let modelId: String
        let modelName: String?
        let providerName: String?
        let inputModalities: [String]?
        let outputModalities: [String]?
        let inferenceTypesSupported: [String]?
        let modelLifecycle: Lifecycle?
        let converse: Converse?
    }

    struct InferenceProfileList: Decodable {
        let inferenceProfileSummaries: [InferenceProfileSummary]?
        let nextToken: String?
    }

    struct InferenceProfileSummary: Decodable {
        struct Model: Decodable {
            let modelArn: String?
        }

        let inferenceProfileId: String
        let status: String?
        let models: [Model]?
    }
}

// MARK: - Details cache

/// The details of the last fetched Bedrock list, keyed by invocation ID, so the reasoning
/// picker and requests can read each model's schema. The ID list itself lives in
/// `LiveModelListStore` with the other providers' lists.
final class BedrockModelDetailsStore: @unchecked Sendable {
    static let shared = BedrockModelDetailsStore()

    private struct Snapshot: Codable {
        let region: String
        let models: [BedrockModelInfo]
    }

    private static let defaultsKey = "AmazonBedrockLiveModelDetails"
    private let lock = NSLock()
    private let defaults = UserDefaults.standard
    private var snapshot: Snapshot?
    private var byID: [String: BedrockModelInfo] = [:]
    private var didLoad = false

    private init() {}

    func save(_ models: [BedrockModelInfo], region: String) {
        let snapshot = Snapshot(region: BedrockRegion.normalized(region), models: models)
        lock.withLock {
            set(snapshot)
            if let data = try? JSONEncoder().encode(snapshot) {
                defaults.set(data, forKey: Self.defaultsKey)
            }
        }
    }

    /// The details for a listed model in the current Region, or `nil` for custom IDs, models
    /// from another Region's list, and when no live list is cached.
    func info(forInvocationID invocationID: String) -> BedrockModelInfo? {
        guard LiveModelListStore.shared.models(forProviderKey: AIProvider.bedrock.rawValue) != nil else {
            return nil
        }
        let id = invocationID.trimmingCharacters(in: .whitespacesAndNewlines)
        let region = BedrockRegion.current
        return lock.withLock {
            loadIfNeeded()
            guard snapshot?.region == region else { return nil }
            return byID[id]
        }
    }

    func clear() {
        lock.withLock {
            snapshot = nil
            byID = [:]
            didLoad = true
            defaults.removeObject(forKey: Self.defaultsKey)
        }
    }

    private func loadIfNeeded() {
        guard !didLoad else { return }
        didLoad = true
        guard let data = defaults.data(forKey: Self.defaultsKey),
            let saved = try? JSONDecoder().decode(Snapshot.self, from: data)
        else {
            return
        }
        set(saved)
    }

    private func set(_ snapshot: Snapshot) {
        self.snapshot = snapshot
        didLoad = true
        byID = Dictionary(snapshot.models.map { ($0.invocationID, $0) }, uniquingKeysWith: { first, _ in first })
    }
}

extension BedrockModelCatalog {
    /// Drops the cached list when the Region changes, since model availability and IDs differ
    /// per Region, and asks for a new list for the new Region.
    static func regionDidChange() {
        BedrockModelDetailsStore.shared.clear()
        LiveModelListStore.shared.clear(forProviderKey: AIProvider.bedrock.rawValue)
        // AIService refetches a missing list for a saved key on this notification.
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .aiProviderKeyChanged, object: nil)
        }
    }
}
