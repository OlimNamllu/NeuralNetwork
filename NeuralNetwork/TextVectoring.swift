//
//  CharacterToVector.swift
//  NeuralNetwork
//
//  Created by Milo Ullman on 5/8/26.
//

import Foundation

extension String {
    func rightAligned(toLength length: Int, padCharacter: String = " ") -> String {
        if self.count >= length {
            return String(self.suffix(length))
        }
        let neededPad = length - self.count
        return String(repeating: padCharacter, count: neededPad) + self
    }
}



let alphabetDictionary: [Character: Int] = Dictionary(
    uniqueKeysWithValues: vocab.enumerated().map { ($1, $0) }
)



func characterToVector(_ character: Character) -> [CGFloat]? {
    guard let index = alphabetDictionary[character] else { return nil }
    let size = alphabetDictionary.count
    var vector = Array(repeating: CGFloat(0), count: size)
    vector[index] = 1
    return vector
}


func generateOneHotVectors(for string: String) -> [CGFloat] {
    let context = string.rightAligned(toLength: textContextSize)
    
    var oneHotVecotors: [CGFloat] = []
    
    for character in context {
        if let vector = characterToVector(character) {
            oneHotVecotors.append(contentsOf: vector)
        }
    }
    
    return oneHotVecotors
}
