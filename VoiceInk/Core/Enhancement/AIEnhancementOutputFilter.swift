import Foundation

struct AIEnhancementOutputFilter {
    private static let reasoningTags = ["thinking", "think", "reasoning"]

    static func filter(_ text: String) -> String {
        var processedText = text

        for tag in reasoningTags {
            // Complete blocks: <think>…</think>
            processedText = replacing(#"(?s)<\#(tag)>.*?</\#(tag)>"#, in: processedText)
            // Reasoning whose opening tag was in the prompt template: "…</think>answer"
            processedText = replacing(#"(?s)\A.*</\#(tag)>"#, in: processedText)
            // Reasoning cut off before its closing tag: "answer<think>…"
            processedText = replacing(#"(?s)<\#(tag)>.*\z"#, in: processedText)
        }

        return processedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replacing(_ pattern: String, in text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
    }
}
