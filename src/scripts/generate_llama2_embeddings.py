import argparse
import numpy as np
from llama_cpp import Llama
import sys
import os
import jsonlines


def testGenEmbedds():
    parser = argparse.ArgumentParser()
    # Default model downloaded and saved in local folder
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/13b/ggml-model.bin",
    )
    args = parser.parse_args()
    llm = Llama(model_path=args.model, embedding=True)
    print(llm.create_embedding("Hello world!"))


def cosine_similarity(vector1, vector2):
    # Calculates the cosine similarity of two vectors.
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity


def loopThrgandGetEmbedd(loc_json):
    # add trained model
    llm = Llama(
        model_path="models_lamma/codellama-7b.Q6_K.gguf",
        embedding=True,
    )

    getlistofjsonl = os.listdir(loc_json)
    for eachJSONL in getlistofjsonl:
        print("Saving embeddings for: ", eachJSONL)
        dictHere = dict()
        with jsonlines.open(loc_json + "/" + eachJSONL, "r") as json_file:
            for eachLine in json_file:
                dictHere[eachLine["docID"]] = llm.create_embedding(
                    eachLine["reviews"],
                )
                # print(embeddings)
        np.save(eachJSONL[:-6] + ".npy", dictHere)
        sys.exit(0)


loopThrgandGetEmbedd(
    "C:/Users/asa/Desktop/Projects/22Math_recSys/data/Final_All/allWithComplexity",
)
