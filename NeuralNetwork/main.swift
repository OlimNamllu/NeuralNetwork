//
//  main.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 3/8/26.
//


import Foundation

// MARK: - Saving Parameters

func writeParameters(_ params: [Double], delete: Bool = false) {
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

var scores: [Double] = []

let path2 = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/Scores.swift"
let url2 = URL(fileURLWithPath: path2)

func writeScores(_ scores: [Double], title: String) {
    let swift = "let scores\(title): [Double] = \(scores) \n \n"
    
    if let fileHandle = try? FileHandle(forWritingTo: url2) {
        defer { try? fileHandle.close() } // Always close the file stream
        let _ = try? fileHandle.seekToEnd()
        if let data = swift.data(using: .utf8) {
            try? fileHandle.write(contentsOf: data)
        }
    }
}

var tim = "/Users/tim/Documents/neural_network/NeuralNetwork/"
var milo = "/Users/miloullman/Desktop/NeuralNetwork/NeuralNetwork/"

func getPath(filename: String) -> String {
    return milo + filename
}

func fetchParameters() -> [Double] {
    let decoder = JSONDecoder()
    let url = URL(filePath: getPath(filename: "Parameters.json"))
    
    let data = try! Data(contentsOf: url)
    var result = [Double]()
    
    do {
        result = try decoder.decode([Double].self, from: data)
    } catch {
        print("JSON decoder is selling")
    }
    
    return result
}



// MARK: - EXECUTE

let trainingData:
