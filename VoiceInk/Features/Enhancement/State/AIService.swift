import Combine
import Foundation
import LLMkit

enum AIProvider: String, CaseIterable, Sendable {
    case cerebras = "Cerebras"
    case groq = "Groq"
    case gemini = "Gemini"
    case anthropic = "Anthropic"
    case openAI = "OpenAI"
    case openRouter = "OpenRouter"
    case mistral = "Mistral"
    case bedrock = "Amazon Bedrock"
    case elevenLabs = "ElevenLabs"
    case deepgram = "Deepgram"
    case soniox = "Soniox"
    case speechmatics = "Speechmatics"
    case assemblyAI = "AssemblyAI"
    case voiceInkRefine = "VoiceInk Refine"
    case ollama = "Ollama"
    case localCLI = "Local CLI"
    case custom = "Custom"

    var baseURL: String {
        switch self {
        case .cerebras:
            return "https://api.cerebras.ai/v1/chat/completions"
        case .groq:
            return "https://api.groq.com/openai/v1/chat/completions"
        case .gemini:
            return "https://generativelanguage.googleapis.com/v1/interactions"
        case .anthropic:
            return "https://api.anthropic.com/v1/messages"
        case .openAI:
            return "https://api.openai.com/v1/chat/completions"
        case .openRouter:
            return "https://openrouter.ai/api/v1/chat/completions"
        case .mistral:
            return "https://api.mistral.ai/v1/chat/completions"
        case .bedrock:
            return BedrockRegion.runtimeEndpoint(for: BedrockRegion.current)
        case .elevenLabs:
            return "https://api.elevenlabs.io/v1/speech-to-text"
        case .deepgram:
            return "https://api.deepgram.com/v1/listen"
        case .soniox:
            return "https://api.soniox.com/v1"
        case .speechmatics:
            return "https://asr.api.speechmatics.com/v2"
        case .assemblyAI:
            return "https://api.assemblyai.com/v2/transcript"
        case .voiceInkRefine:
            return ""
        case .ollama:
            return UserDefaults.standard.string(forKey: "ollamaBaseURL") ?? "http://localhost:11434"
        case .localCLI:
            return ""
        case .custom:
            return UserDefaults.standard.string(forKey: "customProviderBaseURL") ?? ""
        }
    }

    var defaultModel: String {
        switch self {
        case .cerebras:
            return "qwen-3.8-27b"
        case .groq:
            return "qwen/qwen3.8-27b"
        case .gemini:
            return "gemini-3.8-flash"
        case .anthropic:
            return "claude-sonnet-5-5"
        case .openAI:
            return "gpt-6-luna"
        case .mistral:
            return "mistral-small-latest"
        case .bedrock:
            return "global.amazon.nova-2-lite-v1:0"
        case .elevenLabs:
            return "scribe_v2"
        case .deepgram:
            return "whisper-1"
        case .soniox:
            return "stt-async-v6"
        case .speechmatics:
            return "speechmatics-enhanced"
        case .assemblyAI:
            return "universal-3-5-pro"
        case .voiceInkRefine:
            return VoiceInkRefineService.modelName
        case .ollama:
            return UserDefaults.standard.string(forKey: "ollamaSelectedModel") ?? "mistral"
        case .localCLI:
            return "local-cli"
        case .custom:
            return CustomAIProviderManager.shared.defaultModelName
        case .openRouter:
            return "qwen/qwen3.8-27b"
        }
    }

    var availableModels: [String] {
        switch self {
        case .cerebras:
            return [
                "qwen-3.8-27b",
                "gpt-oss-120b",
            ]
        case .groq:
            return [
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ]
        case .gemini:
            // Offline fallback; AIService replaces it with the live list once a key is set.
            return [
                "gemini-3.8-flash",
                "gemini-4-hydrogen",
                "gemini-4-argon",
                "gemini-4-carbon",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite",
            ]
        case .anthropic:
            return [
                "claude-sonnet-5-5",
                "claude-haiku-5-5",
            ]
        case .openAI:
            // Offline fallback; AIService replaces it with the live list once a key is set.
            return [
                "gpt-6-luna",
                "gpt-6.1-terra",
                "gpt-6-sol",
                "gpt-5.6-luna",
                "gpt-5.6-terra",
                "gpt-5.6-sol",
                "gpt-5.5",
                "gpt-5.4",
                "gpt-5.4-mini",
            ]
        case .mistral:
            return [
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-4",
                "mistral-large-latest",
            ]
        case .bedrock:
            // Each one answered VoiceInk's Default prompt in under 4 s in live tests on 2026-11-18.
            // global. profiles work from most commercial Regions; the others are in-Region model IDs.
            return [
                "global.amazon.nova-2-lite-v1:0",
                "global.anthropic.claude-sonnet-5-5",
                "global.openai.gpt-6-luna",
                "qwen.qwen3-next-80b-a3b",
                "openai.gpt-oss-120b-1:0",
            ]
        case .elevenLabs:
            return ["scribe_v2", "scribe_v2_medical"]
        case .deepgram:
            return ["whisper-1"]
        case .soniox:
            return ["stt-async-v6"]
        case .speechmatics:
            return ["speechmatics-enhanced"]
        case .assemblyAI:
            return ["universal-3-5-pro"]
        case .voiceInkRefine:
            return [VoiceInkRefineService.modelName]
        case .ollama:
            return []
        case .localCLI:
            return []
        case .custom:
            return CustomAIProviderManager.shared.availableModelNames
        case .openRouter:
            return []
        }
    }

    var requiresAPIKey: Bool {
        switch self {
        case .voiceInkRefine, .ollama, .localCLI:
            return false
        default:
            return true
        }
    }

    var supportsEnhancement: Bool {
        switch self {
        case .elevenLabs, .deepgram, .soniox, .speechmatics, .assemblyAI:
            return false
        default:
            return true
        }
    }

    /// Providers whose enhancement model list is fetched from their models endpoint.
    var fetchesLiveModelList: Bool {
        switch self {
        case .gemini, .openAI:
            return true
        case .bedrock:
            return true
        default:
            return false
        }
    }

    var supportsCustomModelID: Bool {
        switch self {
        case .cerebras, .groq, .gemini, .anthropic, .openAI, .mistral:
            return true
        case .bedrock:
            return true
        default:
            return false
        }
    }
}

struct OllamaRefreshResult {
    let models: [OllamaModel]
    let errorMessage: String?
}

class AIService: ObservableObject {
    @Published var apiKey: String = ""
    @Published var isAPIKeyValid: Bool = false
    @Published var customBaseURL: String = UserDefaults.standard.string(forKey: "customProviderBaseURL") ?? "" {
        didSet {
            userDefaults.set(customBaseURL, forKey: "customProviderBaseURL")
        }
    }
    @Published var customModel: String = UserDefaults.standard.string(forKey: "customProviderModel") ?? "" {
        didSet {
            userDefaults.set(customModel, forKey: "customProviderModel")
        }
    }
    @Published var selectedProvider: AIProvider {
        didSet {
            userDefaults.set(selectedProvider.rawValue, forKey: "selectedAIProvider")
            if selectedProvider.requiresAPIKey {
                if let savedKey = APIKeyManager.shared.getAPIKey(forProvider: selectedProvider.rawValue) {
                    self.apiKey = savedKey
                    self.isAPIKeyValid = true
                } else {
                    self.apiKey = ""
                    self.isAPIKeyValid = false
                }
            } else {
                self.apiKey = ""
                if selectedProvider == .localCLI {
                    self.isAPIKeyValid = localCLIService.isConfigured
                } else if selectedProvider == .voiceInkRefine {
                    self.isAPIKeyValid = voiceInkRefineService.isAvailableInModes
                } else {
                    self.isAPIKeyValid = true
                }
                if selectedProvider == .ollama {
                    Task {
                        await refreshOllamaAvailability()
                    }
                }
            }
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }
    }

    @Published private var selectedModels: [AIProvider: String] = [:]
    private let userDefaults = UserDefaults.standard
    let voiceInkRefineService = VoiceInkRefineService.shared
    private lazy var ollamaService = OllamaService()
    private lazy var localCLIService = LocalCLIService()
    private var apiKeyChangeObserver: NSObjectProtocol?
    private var voiceInkRefineObserver: AnyCancellable?

    @Published private var openRouterModels: [String] = []
    @Published private var openRouterModelCatalog: [OpenRouterModel] = []
    private var isOpenRouterCatalogRefreshing = false
    @Published private(set) var refreshingLiveModelProviders: Set<AIProvider> = []
    private var liveModelListObserver: NSObjectProtocol?
    private static let liveModelListRefreshInterval: TimeInterval = 24 * 60 * 60
    @Published private(set) var isOllamaRefreshing = false

    var connectedProviders: [AIProvider] {
        AIProvider.allCases.filter { provider in
            guard provider.supportsEnhancement else {
                return false
            }

            if provider == .custom {
                return CustomAIProviderManager.shared.hasConfiguredModels
            } else if provider == .voiceInkRefine {
                return voiceInkRefineService.isAvailableInModes
            } else if provider == .ollama {
                return ollamaService.isConnected
            } else if provider == .localCLI {
                return localCLIService.isConfigured
            } else if provider.requiresAPIKey {
                return APIKeyManager.shared.hasAPIKey(forProvider: provider.rawValue)
            }
            return false
        }
    }

    var currentModel: String {
        if selectedProvider == .voiceInkRefine {
            return selectedProvider.defaultModel
        }

        if let selectedModel = selectedModels[selectedProvider],
            !selectedModel.isEmpty,
            (selectedProvider.supportsCustomModelID || selectedProvider == .ollama
                || availableModels.contains(selectedModel))
        {
            return selectedModel
        }
        return selectedProvider.defaultModel
    }

    func selectedModel(for provider: AIProvider) -> String {
        if provider == .voiceInkRefine {
            return provider.defaultModel
        }

        if let selectedModel = selectedModels[provider], !selectedModel.isEmpty {
            return selectedModel
        }
        return provider.defaultModel
    }

    func customModelID(for provider: AIProvider) -> String {
        guard provider.supportsCustomModelID else { return "" }
        let key = "\(provider.rawValue)CustomModelID"
        if let savedModel = userDefaults.string(forKey: key), !savedModel.isEmpty {
            return savedModel
        }
        let selectedModel = selectedModel(for: provider)
        return availableModels(for: provider).contains(selectedModel) ? "" : selectedModel
    }

    var availableModels: [String] {
        availableModels(for: selectedProvider)
    }

    var localCLICommandTemplate: String {
        localCLIService.commandTemplate
    }

    var localCLITemplateSelection: LocalCLITemplate {
        localCLIService.selectedTemplate
    }

    var localCLITimeoutSeconds: Double {
        localCLIService.timeoutSeconds
    }

    func availableModels(for provider: AIProvider) -> [String] {
        if provider == .ollama {
            return ollamaService.availableModels.map { $0.name }
        } else if provider == .openRouter {
            return openRouterModels
        } else if provider == .custom {
            return CustomAIProviderManager.shared.availableModelNames
        } else if provider.fetchesLiveModelList,
            let models = LiveModelListStore.shared.models(forProviderKey: provider.rawValue)
        {
            return models
        }
        return provider.availableModels
    }

    init() {
        if userDefaults.string(forKey: "selectedAIProvider") == "GROQ" {
            userDefaults.set("Groq", forKey: "selectedAIProvider")
        }
        GeminiModelMigration.migrateSavedSelection(in: userDefaults)

        if let savedProvider = userDefaults.string(forKey: "selectedAIProvider"),
            let provider = AIProvider(rawValue: savedProvider)
        {
            self.selectedProvider = provider
        } else {
            self.selectedProvider = .gemini
        }

        if selectedProvider.requiresAPIKey {
            if let savedKey = APIKeyManager.shared.getAPIKey(forProvider: selectedProvider.rawValue) {
                self.apiKey = savedKey
                self.isAPIKeyValid = true
            }
        } else {
            if selectedProvider == .localCLI {
                self.isAPIKeyValid = localCLIService.isConfigured
            } else if selectedProvider == .voiceInkRefine {
                self.isAPIKeyValid = voiceInkRefineService.isAvailableInModes
            } else {
                self.isAPIKeyValid = true
            }
        }

        CerebrasModelMigration.migrateSavedSelection(in: userDefaults)
        loadSavedModelSelections()
        loadSavedOpenRouterModels()

        // Observe installation state without forwarding every progress update to the entire scene.
        voiceInkRefineObserver = voiceInkRefineService.$isDownloaded
            .removeDuplicates()
            .dropFirst()
            .sink { [weak self] _ in
                DispatchQueue.main.async {
                    guard let self else { return }

                    if self.selectedProvider == .voiceInkRefine {
                        let isAvailable = self.voiceInkRefineService.isAvailableInModes
                        if self.isAPIKeyValid != isAvailable {
                            self.isAPIKeyValid = isAvailable
                        }
                    }

                    self.objectWillChange.send()
                }
            }

        apiKeyChangeObserver = NotificationCenter.default.addObserver(
            forName: .aiProviderKeyChanged,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                self.reloadSelectedProviderConfiguration()
                self.syncLiveProviderModelsWithAPIKeys()
            }
        }

        liveModelListObserver = NotificationCenter.default.addObserver(
            forName: LiveModelListStore.didChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            DispatchQueue.main.async {
                self?.objectWillChange.send()
            }
        }
    }

    deinit {
        if let apiKeyChangeObserver {
            NotificationCenter.default.removeObserver(apiKeyChangeObserver)
        }
        if let liveModelListObserver {
            NotificationCenter.default.removeObserver(liveModelListObserver)
        }
        voiceInkRefineObserver?.cancel()
    }

    private func reloadSelectedProviderConfiguration() {
        if selectedProvider == .custom {
            customBaseURL = userDefaults.string(forKey: "customProviderBaseURL") ?? ""
            customModel = userDefaults.string(forKey: "customProviderModel") ?? ""
        }

        let selectedModelKey = "\(selectedProvider.rawValue)SelectedModel"
        if selectedProvider == .voiceInkRefine {
            selectedModels[selectedProvider] = selectedProvider.defaultModel
        } else if let savedModel = userDefaults.string(forKey: selectedModelKey), !savedModel.isEmpty {
            selectedModels[selectedProvider] = savedModel
        }

        if selectedProvider.requiresAPIKey {
            if let savedKey = APIKeyManager.shared.getAPIKey(forProvider: selectedProvider.rawValue) {
                apiKey = savedKey
                isAPIKeyValid = true
            } else {
                apiKey = ""
                isAPIKeyValid = false
            }
        } else {
            apiKey = ""
            if selectedProvider == .localCLI {
                isAPIKeyValid = localCLIService.isConfigured
            } else if selectedProvider == .voiceInkRefine {
                isAPIKeyValid = voiceInkRefineService.isAvailableInModes
            } else {
                isAPIKeyValid = true
            }
        }
    }

    private func loadSavedModelSelections() {
        for provider in AIProvider.allCases {
            if provider == .voiceInkRefine {
                selectedModels[provider] = provider.defaultModel
                continue
            }

            let key = "\(provider.rawValue)SelectedModel"
            if let savedModel = userDefaults.string(forKey: key), !savedModel.isEmpty {
                selectedModels[provider] = savedModel
            }
        }
    }

    private func loadSavedOpenRouterModels() {
        if let catalog = OpenRouterCatalogStore.shared.models(for: .enhancement) {
            openRouterModelCatalog = catalog
            openRouterModels = catalog.filter(isOpenRouterEnhancementModel).map(\.id)
            reconcileOpenRouterSelection(selectInitialIfNeeded: true)
            return
        }

        openRouterModels = OpenRouterCatalogStore.shared.legacyEnhancementModelIDs
    }

    private func saveOpenRouterModels() {
        OpenRouterCatalogStore.shared.saveLegacyEnhancementModelIDs(openRouterModels)
        try? OpenRouterCatalogStore.shared.save(openRouterModelCatalog, for: .enhancement)
    }

    private func isOpenRouterEnhancementModel(_ model: OpenRouterModel) -> Bool {
        guard let architecture = model.architecture else { return true }
        return architecture.inputModalities.contains("text")
            && architecture.outputModalities.contains("text")
    }

    @discardableResult
    private func reconcileOpenRouterSelection(selectInitialIfNeeded: Bool = false) -> Bool {
        let selected = selectedModels[.openRouter]
        if let selected, openRouterModels.contains(selected) { return false }
        guard selected != nil || (selectInitialIfNeeded && selectedProvider == .openRouter) else { return false }

        if let replacement = openRouterModels.first(where: { $0 == AIProvider.openRouter.defaultModel })
            ?? openRouterModels.first
        {
            selectedModels[.openRouter] = replacement
            userDefaults.set(replacement, forKey: "OpenRouterSelectedModel")
        } else {
            selectedModels.removeValue(forKey: .openRouter)
            userDefaults.removeObject(forKey: "OpenRouterSelectedModel")
        }
        return selected != selectedModels[.openRouter]
    }

    func selectModel(_ model: String) {
        selectModel(model, for: selectedProvider)
    }

    func selectModel(_ model: String, for provider: AIProvider) {
        let resolvedInput = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !resolvedInput.isEmpty else { return }

        if provider == .custom {
            guard CustomAIProviderManager.shared.applyConfiguration(forModel: resolvedInput) else { return }
        }

        let resolvedModel = provider == .voiceInkRefine ? provider.defaultModel : resolvedInput
        selectedModels[provider] = resolvedModel
        let key = "\(provider.rawValue)SelectedModel"
        userDefaults.set(resolvedModel, forKey: key)
        if provider.supportsCustomModelID, !availableModels(for: provider).contains(resolvedModel) {
            userDefaults.set(resolvedModel, forKey: "\(provider.rawValue)CustomModelID")
        }

        if provider == .ollama {
            updateSelectedOllamaModel(resolvedModel)
        } else if provider == .custom {
            reloadSelectedProviderConfiguration()
        }

        objectWillChange.send()
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }

    func saveAPIKey(_ key: String, completion: @escaping (Bool, String?) -> Void) {
        guard selectedProvider.requiresAPIKey else {
            completion(true, nil)
            return
        }

        verifyAPIKey(key) { [weak self] isValid, errorMessage in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if isValid {
                    self.apiKey = key
                    self.isAPIKeyValid = true
                    APIKeyManager.shared.saveAPIKey(key, forProvider: self.selectedProvider.rawValue)
                    NotificationCenter.default.post(name: .aiProviderKeyChanged, object: nil)
                    NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
                } else {
                    self.isAPIKeyValid = false
                }
                completion(isValid, errorMessage)
            }
        }
    }

    func verifyAPIKey(_ key: String, completion: @escaping (Bool, String?) -> Void) {
        guard selectedProvider.requiresAPIKey else {
            completion(true, nil)
            return
        }

        Task {
            let result = await verifyAPIKey(
                key,
                for: selectedProvider,
                model: currentModel
            )
            DispatchQueue.main.async {
                completion(result.isValid, result.errorMessage)
            }
        }
    }

    func verifyAPIKey(_ key: String, for provider: AIProvider, model: String? = nil) async -> (
        isValid: Bool, errorMessage: String?
    ) {
        guard provider.requiresAPIKey else {
            return (true, nil)
        }

        let verificationModel = model ?? selectedModel(for: provider)
        let result: (isValid: Bool, errorMessage: String?)

        switch provider {
        case .anthropic:
            result = await AnthropicLLMClient.verifyAPIKey(key)
        case .elevenLabs:
            result = await ElevenLabsClient.verifyAPIKey(key)
        case .deepgram:
            result = await DeepgramClient.verifyAPIKey(key)
        case .mistral:
            result = await MistralTranscriptionClient.verifyAPIKey(key)
        case .soniox:
            result = await SonioxClient.verifyAPIKey(key)
        case .speechmatics:
            result = await SpeechmaticsClient.verifyAPIKey(key)
        case .assemblyAI:
            result = await AssemblyAIClient.verifyAPIKey(key)
        case .openRouter:
            result = await OpenRouterClient.verifyAPIKey(key)
        case .openAI:
            result = await OpenAILLMClient.verifyAPIKey(key)
        case .gemini:
            result = await GeminiLLMClient.verifyAPIKey(key)
        case .bedrock:
            result = await BedrockConverseClient.verifyAPIKey(
                key,
                region: BedrockRegion.current,
                modelID: verificationModel
            )
        case .cerebras:
            // GET /v1/models checks the key without depending on a model ID.
            result = await OpenAITranscriptionClient.verifyAPIKey(
                baseURL: URL(string: "https://api.cerebras.ai")!,
                apiKey: key
            )
        default:
            guard let baseURL = URL(string: provider.baseURL) else {
                return (false, "Invalid or missing base URL configuration")
            }
            result = await OpenAILLMClient.verifyAPIKey(
                baseURL: baseURL,
                apiKey: key,
                model: verificationModel
            )
        }

        return result
    }

    func clearAPIKey() {
        guard selectedProvider.requiresAPIKey else { return }

        apiKey = ""
        isAPIKeyValid = false
        APIKeyManager.shared.deleteAPIKey(forProvider: selectedProvider.rawValue)
        NotificationCenter.default.post(name: .aiProviderKeyChanged, object: nil)
    }

    func checkOllamaConnection(completion: @escaping (Bool) -> Void) {
        Task { [weak self] in
            guard let self = self else { return }
            _ = await self.refreshOllamaAvailability()
            await MainActor.run {
                completion(self.ollamaService.isConnected)
            }
        }
    }

    func refreshOllamaAvailabilityInBackground() {
        Task { [weak self] in
            guard let self else { return }
            _ = await self.refreshOllamaAvailability()
        }
    }

    @MainActor
    @discardableResult
    func refreshOllamaConnectionAndModels() async -> [OllamaModel] {
        let result = await refreshOllamaAvailability()
        return result.models
    }

    @MainActor
    func refreshOllamaAvailability() async -> OllamaRefreshResult {
        guard !isOllamaRefreshing else {
            return OllamaRefreshResult(models: ollamaService.availableModels, errorMessage: nil)
        }

        isOllamaRefreshing = true
        defer {
            isOllamaRefreshing = false
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }

        let result = await ollamaService.refreshConnectionAndModels()
        switch result {
        case .success(let models):
            return OllamaRefreshResult(models: models, errorMessage: nil)
        case .failure(let error):
            return OllamaRefreshResult(models: [], errorMessage: ollamaErrorMessage(for: error))
        }
    }

    private func ollamaErrorMessage(for error: Error) -> String {
        if let llmKitError = error as? LLMKitError {
            return ollamaErrorMessage(for: llmKitError)
        }

        if let localAIError = error as? LocalAIError,
            let errorDescription = localAIError.errorDescription
        {
            return errorDescription
        }

        let nsError = error as NSError
        var details = [nsError.localizedDescription]

        if let failingURL = nsError.userInfo["NSErrorFailingURLKey"] as? URL {
            details.append("URL: \(failingURL.absoluteString)")
        } else if let failingURLString = nsError.userInfo["NSErrorFailingURLStringKey"] as? String {
            details.append("URL: \(failingURLString)")
        }

        details.append("Code: \(nsError.domain) \(nsError.code)")

        if let underlyingError = nsError.userInfo[NSUnderlyingErrorKey] as? NSError {
            details.append("Underlying: \(underlyingError.localizedDescription)")
            details.append("Underlying code: \(underlyingError.domain) \(underlyingError.code)")
        }

        if let streamErrorCode = nsError.userInfo["_kCFStreamErrorCodeKey"] {
            details.append("Network code: \(streamErrorCode)")
        }

        return details.joined(separator: "\n")
    }

    private func ollamaErrorMessage(for error: LLMKitError) -> String {
        switch error {
        case .httpError(let statusCode, let message):
            let trimmedMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedMessage.isEmpty else { return "HTTP \(statusCode)" }
            return "HTTP \(statusCode): \(trimmedMessage)"
        default:
            return error.localizedDescription
        }
    }

    func enhanceWithOllama(
        text: String,
        systemPrompt: String,
        model: String? = nil,
        think: Bool = false,
        timeout: TimeInterval = 30,
        retryOnTimeout: Bool = false
    ) async throws -> String {
        try await ollamaService.enhance(
            text,
            withSystemPrompt: systemPrompt,
            model: model,
            think: think,
            timeout: timeout,
            retryOnTimeout: retryOnTimeout
        )
    }

    func enhanceWithVoiceInkRefine(transcript: String, vocabulary: RecordingVocabulary) async throws -> String {
        try await voiceInkRefineService.enhance(transcript: transcript, vocabulary: vocabulary)
    }

    func updateOllamaBaseURL(_ newURL: String) {
        ollamaService.baseURL = newURL
        userDefaults.set(newURL, forKey: "ollamaBaseURL")
    }

    func updateSelectedOllamaModel(_ modelName: String) {
        ollamaService.selectedModel = modelName
        userDefaults.set(modelName, forKey: "ollamaSelectedModel")
    }

    func loadLocalCLITemplate(_ template: LocalCLITemplate) {
        localCLIService.loadTemplate(template)
        refreshLocalCLIConfigurationState()
    }

    func updateLocalCLICommandTemplate(_ command: String) {
        localCLIService.commandTemplate = command
        refreshLocalCLIConfigurationState()
    }

    func updateLocalCLITimeoutSeconds(_ timeout: Double) {
        localCLIService.timeoutSeconds = timeout
        refreshLocalCLIConfigurationState()
    }

    func enhanceWithLocalCLI(systemPrompt: String, userPrompt: String) async throws -> String {
        try await localCLIService.enhance(systemPrompt: systemPrompt, userPrompt: userPrompt)
    }

    private func refreshLocalCLIConfigurationState() {
        if selectedProvider == .localCLI {
            isAPIKeyValid = localCLIService.isConfigured
        }
        objectWillChange.send()
        NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
    }

    func openRouterModelMetadata(for modelName: String) -> OpenRouterModel? {
        openRouterModelCatalog.first(where: { $0.id == modelName })
    }

    func reasoningModelInfo(for provider: AIProvider, model: String) -> ReasoningModelInfo {
        switch provider {
        case .openRouter:
            return ReasoningModelInfo(openRouterModel: openRouterModelMetadata(for: model))
        case .ollama:
            return ReasoningModelInfo(ollamaSupportsThinking: ollamaService.thinkingSupport[model])
        case .bedrock:
            return ReasoningModelInfo(bedrockModel: BedrockModelDetailsStore.shared.info(forInvocationID: model))
        default:
            return ReasoningModelInfo()
        }
    }

    @MainActor
    func fetchOpenRouterModelsIfNeededForMigration() async {
        guard openRouterModelCatalog.isEmpty,
            APIKeyManager.shared.hasAPIKey(forProvider: AIProvider.openRouter.rawValue)
        else {
            return
        }

        await fetchOpenRouterModels()
    }

    @MainActor
    func fetchOpenRouterModels() async {
        guard !isOpenRouterCatalogRefreshing else { return }
        isOpenRouterCatalogRefreshing = true
        defer { isOpenRouterCatalogRefreshing = false }

        do {
            let catalog = try await OpenRouterClient.fetchModelCatalog()
            openRouterModelCatalog = catalog
            openRouterModels = catalog.filter(isOpenRouterEnhancementModel).map(\.id)
            saveOpenRouterModels()
            if reconcileOpenRouterSelection(selectInitialIfNeeded: true) {
                NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
            }
            objectWillChange.send()
        } catch {
            // Keep the last successful catalog during transient OpenRouter failures.
        }
    }

    // MARK: - Live Gemini and OpenAI model lists

    /// Keeps the cached live lists tied to a saved key: drops a list when its key is removed,
    /// and fetches one for a newly added key.
    private func syncLiveProviderModelsWithAPIKeys() {
        var didChange = false
        for provider in AIProvider.allCases where provider.fetchesLiveModelList {
            if APIKeyManager.shared.hasAPIKey(forProvider: provider.rawValue) {
                if LiveModelListStore.shared.models(forProviderKey: provider.rawValue) == nil {
                    Task { @MainActor [weak self] in
                        _ = await self?.fetchLiveModels(for: provider)
                    }
                }
            } else if LiveModelListStore.shared.clear(forProviderKey: provider.rawValue) {
                didChange = true
            }
        }
        if didChange {
            NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
        }
    }

    /// Refreshes the Gemini and OpenAI model lists at launch when a key is saved and the
    /// cached list is missing or older than a day.
    @MainActor
    func refreshLiveProviderModelsIfNeeded() async {
        for provider in AIProvider.allCases where provider.fetchesLiveModelList {
            guard APIKeyManager.shared.hasAPIKey(forProvider: provider.rawValue) else { continue }
            let store = LiveModelListStore.shared
            if store.models(forProviderKey: provider.rawValue) != nil,
                let fetchedAt = store.fetchedAt(forProviderKey: provider.rawValue),
                Date().timeIntervalSince(fetchedAt) < Self.liveModelListRefreshInterval
            {
                continue
            }
            _ = await fetchLiveModels(for: provider)
        }
    }

    func isRefreshingLiveModels(for provider: AIProvider) -> Bool {
        refreshingLiveModelProviders.contains(provider)
    }

    /// Fetches the provider's current models with the saved key. Returns an error message on
    /// failure; the last successful list (or the built-in fallback) stays in place.
    @MainActor
    @discardableResult
    func fetchLiveModels(for provider: AIProvider) async -> String? {
        guard provider.fetchesLiveModelList else { return nil }
        guard let apiKey = APIKeyManager.shared.getAPIKey(forProvider: provider.rawValue), !apiKey.isEmpty else {
            return String(localized: "Add an API key to load the latest models.")
        }
        guard !refreshingLiveModelProviders.contains(provider) else { return nil }
        refreshingLiveModelProviders.insert(provider)
        defer { refreshingLiveModelProviders.remove(provider) }

        // Bedrock lists differ per Region, so a fetch for one Region must not serve another.
        let bedrockRegion = BedrockRegion.current
        let requestScope = provider == .bedrock ? "\(apiKey)\n\(bedrockRegion)" : apiKey

        do {
            let fetched = try await LiveModelListStore.shared.fetch(forProviderKey: provider.rawValue, apiKey: requestScope) {
                switch provider {
                case .gemini:
                    return try await ProviderModelCatalog.fetchGeminiModelIDs(apiKey: apiKey)
                case .bedrock:
                    return try await BedrockModelCatalog.fetchModelIDs(apiKey: apiKey, region: bedrockRegion)
                default:
                    return try await ProviderModelCatalog.fetchOpenAIModelIDs(apiKey: apiKey)
                }
            }

            // The key may have been removed or replaced while the request was in flight.
            guard APIKeyManager.shared.getAPIKey(forProvider: provider.rawValue) == apiKey else { return nil }
            guard provider != .bedrock || BedrockRegion.current == bedrockRegion else {
                // The Region changed mid-fetch, and its own fetch was skipped while this one ran.
                // Fetch again once this call returns and clears the refreshing flag.
                Task { @MainActor [weak self] in
                    _ = await self?.fetchLiveModels(for: provider)
                }
                return nil
            }

            let models = orderedLiveModels(usableLiveModels(fetched, for: provider), for: provider)
            guard !models.isEmpty else {
                return ProviderModelCatalog.CatalogError.noUsableModels.localizedDescription
            }
            if provider == .bedrock {
                BedrockModelDetailsStore.shared.commitStaged(region: bedrockRegion, apiKey: apiKey)
            }
            if LiveModelListStore.shared.save(models, forProviderKey: provider.rawValue) {
                NotificationCenter.default.post(name: .AppSettingsDidChange, object: nil)
            }
            return nil
        } catch {
            // Keep the last successful list during transient failures or when offline.
            return error.localizedDescription
        }
    }

    /// LLMkit always sends `temperature: 0.3`, which OpenAI's reasoning models reject
    /// unless reasoning effort is `none`. So only offer the OpenAI models whose Default
    /// request VoiceInk has made work: `none`, or `temperature: null` for models that
    /// reject `none`.
    private func usableLiveModels(_ models: [String], for provider: AIProvider) -> [String] {
        guard provider == .openAI else { return models }
        return models.filter(ReasoningConfig.isOpenAIDefaultRequestSupported)
    }

    /// Puts the default model first so the picker opens on the recommended choice.
    private func orderedLiveModels(_ models: [String], for provider: AIProvider) -> [String] {
        guard let defaultIndex = models.firstIndex(of: provider.defaultModel) else { return models }
        var ordered = models
        ordered.remove(at: defaultIndex)
        ordered.insert(provider.defaultModel, at: 0)
        return ordered
    }
}
