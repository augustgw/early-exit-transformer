import argparse
import os
from jiwer import process_words, visualize_alignment

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="Name of model dir in model_results (e.g., 'bi-lstm_170_173')")
args = parser.parse_args()

with open(args.model_dir + '/alignment_results.txt', 'w') as outfile:
    outfile.write(args.model_dir + '\n\n--------------------------------\n\n')

    for test_set in ('dev-clean', 'dev-other', 'test-clean', 'test-other'):
        outfile.write(test_set + '\n\n')
        
        with open(test_set + '_expected.txt', 'r') as expected_file:
            expected = expected_file.readlines()

            for layer in ('layer2', 'layer4', 'layer6', 'layer8', 'layer10', 'layer12'):
                outfile.write(layer + '\n')
                with open(args.model_dir + '/' + test_set + '/' + layer + '.txt', 'r') as hypotheses_file:
                    hypotheses = hypotheses_file.readlines()
                    output = process_words(expected, hypotheses)
                    
                    final_str = ""
                    final_str += f"number of sentences: {len(output.alignments)}\n"
                    final_str += f"substitutions={output.substitutions} "
                    final_str += f"deletions={output.deletions} "
                    final_str += f"insertions={output.insertions} "
                    final_str += f"hits={output.hits}\n\n"
                    
                    outfile.write(final_str)
        
        outfile.write('\n--------------------------------\n\n')

