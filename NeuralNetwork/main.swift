//
//  main.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 3/8/26.
//


import Foundation

// MARK: - Saving Parameters

func writeParameters(_ params: [CGFloat], delete: Bool = false) {
    let swift = "\(params) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: URL(filePath: getPath(filename: "Parameters.json"))) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if delete {
            try? fileHandle.truncate(atOffset: 0)
        }
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

var tim = "/Users/tim/Documents/neural_network/NeuralNetwork/"
var milo = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/"

func getPath(filename: String) -> String {
    return (getUser() + filename)
}

func fetchParameters() -> [CGFloat] {
    let decoder = JSONDecoder()
    let url = URL(filePath: getPath(filename: "Parameters.json"))
    
    let data = try! Data(contentsOf: url)
    var result = [CGFloat]()
    
    do {
        result = try decoder.decode([CGFloat].self, from: data)
    } catch {
        print("JSON decoder is selling")
    }
    
    return result
}



// MARK: - EXECUTE

var network = NeuralNetwork()

network.write(fetchParameters())

network.backprop(iterations: 100000, learnRate: 0.01, batchSize: 56)

print("Training complete. Starting generation")


let seedStart = 13000
let seed = String(sampleData[seedStart..<(seedStart + textContextSize)].map { vocab[$0] })
print(network.generate(from: seed, characters: 500))

writeParameters(network.list(), delete: true)

