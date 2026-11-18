import Foundation

struct AIEnhancementOutputFilter {
    private static let reasoningTags = ["thinking", "think", "reasoning"]

    static func filter(_ text: String) -> String {
        var processedText = text

        for tag in reasoningTags {
            // Complete blocks: <think>…</think>
            processedText = replacing(#"(?s)<\#(tag)>.*?</\#(tag)>"#, in: processedText)
            // A response that opens with reasoning and never closes it (cut off) has
            // no answer. Anchored to the start so a dictated mention of a tag survives.
            processedText = replacing(#"(?s)\A\s*<\#(tag)>.*\z"#, in: processedText)
        }

        return processedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replacing(_ pattern: String, in text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
    }
}
