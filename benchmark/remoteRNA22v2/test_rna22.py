import subprocess

def run_rna22(mi, cts):
    with open('myMirInputFile.txt', 'w') as f:
        f.write(f">mir\n{mi}\n")
    with open('myTranscriptInputFile.txt', 'w') as f:
        f.write(f">cts\n{cts}\n")
    
    subprocess.run(['java', 'RNA22v2'], capture_output=True)
    
    with open('output.txt') as f:
        return f.read()

# Known strong hit
print(run_rna22("UGAGGUAGUAGGUUGUAUAGUU", "AACUAUACAACCUACUACCUCAAACUAUACAACCUACUACCUCA"))
