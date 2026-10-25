"""Check hashes, all saved metrics, and optionally reloaded model probabilities."""
import argparse
from src.progressive_baselines import project_root,output_path,verify,DEFAULT_OUTPUT
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--reload-models',action='store_true')
    a=p.parse_args();root=project_root();result=verify(root,output_path(root,a.output),a.reload_models)
    print(f'Passed {result["check_count"]} progressive checks; no refitting.')
