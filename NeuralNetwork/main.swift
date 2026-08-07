//
//  main.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 3/8/26.
//


import Foundation

// MARK: - Saving Parameters
let path = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Parameters.swift"
let url = URL(fileURLWithPath: path)

func writeParameters(_ params: [CGFloat], title: String) {
    let swift = "let parameters\(title): [CGFloat] = \(params) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: url) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if let data = swift.data(using: .utf8) {
            try? fileHandle.write(contentsOf: data)
        }
    }
}

// MARK: - Saving Parameters

var scores: [CGFloat] = []

let path2 = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Scores.swift"
let url2 = URL(fileURLWithPath: path2)

func writeScores(_ scores: [CGFloat], title: String) {
    let swift = "let scores\(title): [CGFloat] = \(scores) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: url2) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if let data = swift.data(using: .utf8) {
            try? fileHandle.write(contentsOf: data)
        }
    }
}

func regenerateComputerRandom() {
    var computerRandoms: [[CGFloat]] = []
    print("\(humanSamples.count) human samples. Generating \(humanSamples.count) computer samples")
    for _ in 0..<humanSamples.count {
        var number = [CGFloat]()
        
        for _ in 0..<16 {
            let int = Int.random(in: 0..<10)
            
            let cgFloat = CGFloat(int)
            
            number.append(cgFloat)
        }
        
        computerRandoms.append(number)
    }

    let dataPath = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Data.swift"
    let dataurl = URL(fileURLWithPath: dataPath)

    func writeData(_ numbers: [[CGFloat]]) {
        let swift = "let machineSamples: [[CGFloat]] = \(numbers) \n \n"
        
        if let fileHandle = try? FileHandle(forWritingTo: dataurl) {
            defer { try? fileHandle.close() } // Always close the file stream
            let _ = try? fileHandle.seekToEnd()
            if let data = swift.data(using: .utf8) {
                try? fileHandle.write(contentsOf: data)
            }
        }
    }

    writeData(computerRandoms)
}


// MARK: - EXECUTE

//

var homePath1 = "/Users/tim/Documents/neural_network/NeuralNetwork/onestar_corpus.txt"
var homePath2 = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/onestar_corpus.txt"

func get_path() -> String {
    return homePath1
}

var network = NeuralNetwork()
network.write(parametersCXT21_H288_N003)

//network.backprop(iterations: 60000, learnRate: 0.04, batchSize: 56)

print("Training complete. Generating text...")

let seedStart = 7500
let seed = String(sampleData[seedStart..<(seedStart + textContextSize)].map { vocab[$0] })
print(network.generate(from: seed, characters: 300))

//writeParameters(network.list(), title: "CXT21_H288_N003")
 
