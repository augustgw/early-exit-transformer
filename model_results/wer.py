import argparse
import os
from jiwer import wer

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="Name of model dir in model_results (e.g., 'bi-lstm_170_173')")
args = parser.parse_args()

with open(args.model_dir + '/wer_results.txt', 'w') as outfile:
    outfile.write(args.model_dir + '\n\n')

    for test_set in ('dev-clean', 'dev-other', 'test-clean', 'test-other'):
        outfile.write(test_set + '\n')
        
        with open(test_set + '_expected.txt', 'r') as expected_file:
            expected = expected_file.readlines()

            for layer in ('layer2', 'layer4', 'layer6', 'layer8', 'layer10', 'layer12'):
                with open(args.model_dir + '/' + test_set + '/' + layer + '.txt', 'r') as hypotheses_file:
                    hypotheses = hypotheses_file.readlines()
                    outfile.write(layer + ': ' + str(wer(expected, hypotheses)) + '\n')
        
        outfile.write('\n')

