import Foundation
import Testing
@testable import VoiceInkChat

@Suite struct ReasoningTextTests {
    @Test func stripsBlocksAndKeepsAnswers() {
        #expect(ReasoningText.strip("<think>a</think>Answer") == "Answer")
        #expect(ReasoningText.strip("<THINK>a</THINK>\n\nAnswer") == "\n\nAnswer")
        #expect(ReasoningText.strip("A<think>x</think>B") == "AB")
        #expect(ReasoningText.strip("reasoning only</think>Answer") == "Answer")
        #expect(ReasoningText.strip("<think>never closed") == "")
        #expect(ReasoningText.strip("Use a <think> tag in HTML") == "Use a <think> tag in HTML")
        #expect(ReasoningText.strip("Plain answer.") == "Plain answer.")
    }
}

@Suite struct JSONValueTests {
    @Test func roundTripsAndEncodesWholeNumbersAsIntegers() throws {
        let value: JSONValue = ["a": 1, "b": 0.5, "c": .null, "d": [true, "x"], "e": ["f": 1024]]
        let encoded = String(data: try value.encoded(), encoding: .utf8)!
        #expect(encoded == #"{"a":1,"b":0.5,"c":null,"d":[true,"x"],"e":{"f":1024}}"#)
        #expect(try JSONValue(data: Data(encoded.utf8)) == value)
    }
}

@Suite struct HeaderMergeTests {
    @Test func laterLayersWinRegardlessOfCase() {
        let merged = ChatHTTP.mergedHeaders(["Accept": "a", "X-One": "1"], ["accept": "b"], ["ACCEPT": "c"])
        #expect(merged.count == 2)
        #expect(merged["ACCEPT"] == "c")
        #expect(merged["X-One"] == "1")
    }
}
