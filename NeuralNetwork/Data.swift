//
//  Data.swift
//  
//
//  Created by Milo Ullman on 3/8/26.
//

import Foundation


// MARK: - DATA


// Training data: human-style vs. CSPRNG 16-digit integers.
//
// IMPORTANT: each sample is stored as its 16 DIGITS, not as one value.
// Double is a Double; its integer-exact range stops at 2^53 = 9007199254740992,
// so ~10% of 16-digit integers cannot round-trip through Double. Digit vectors
// are both exact and the shape a network actually wants (16 inputs, values 0...9).
// Divide by 9.0 if you want them normalised to 0...1.

/// 64 x 16 - digits produced by hand, no generator. Label 0.


var corpusPath: String {
    getPath(filename: "onestar_corpus.txt")
}

let sampleText = try! String(contentsOf: URL(fileURLWithPath: corpusPath), encoding: .utf8)
    .trimmingCharacters(in: .whitespacesAndNewlines)

let vocab = Array("abcdefghijklmnopqrstuvwxyz ")
let sampleData: [Int] = {
    let mapped = Array(sampleText).map { (ch: Character) -> Int in
        guard let i = vocab.firstIndex(of: ch) else {
            fatalError("Character not in vocab: \(String(reflecting: ch))")
        }
        return i
    }
    print("corpus loaded: \(mapped.count) characters")
    return mapped
}()  // count is 38015
let textContextSize: Int = 16

var sampleSeed: Int {
    return Int.random(in: 4..<36_000)
}

var validationStarts: [Int] {
    return Array(37000..<38000)
}


func encode(startingAt start: Int, noise: Double = 0.00) -> [Double] {
    var input = [Double](repeating: 0, count: textContextSize * 27)
    for slot in 0..<textContextSize {
        var character = sampleData[start + slot]
        if noise > 0 && Double.random(in: 0..<1) < noise {
            character = Int.random(in: 0..<27)
        }
        input[slot * 27 + character] = 1
    }
    return input
}

func makeTarget(startingAt start: Int) -> [Double] {
    var target = [Double](repeating: -0.8, count: 27)
    target[sampleData[start + textContextSize]] = 0.8
    return target
}



