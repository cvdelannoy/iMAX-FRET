# PDB file classification using iMAX-FRET

Run this pipeline using the conda environment `iMAX-FRET_pdb`, as defined in `env_pdb_classification.yaml`:
```bash
conda env create -f env_pdb_classification.yaml
conda activate iMAX-FRET_pdb
```

This pipeline extracts predicted iMAX-FRET fingerprints from PDB files, trains embeddings and classifiers,
and evaluates the accuracy of the classifiers, in N-fold cross validation. Example:
```bash
python run_pipeline_pdb_classifier.py \
   --out-dir path/to/output/dir \
   --up-list uniprot_id_list.txt \                  # subselection of pdb_ids to use
   --buried-txt TYROSINE_BURIED_EXPOSED2.txt \      # accessibility data for these pdb_ids, for a given residue type
   --pdb-dir path/to/pdb_files \                    # path to a directory of pdb files
   --max-nb-proteins 100 \                          # Pick 100 proteins at random
   --nb-folds 10 \                                  # 10-fold cross validation
   --repeats 10                                     # Generate 10 noisy fingerprints per protein

```

Example files for `--buried-txt` and `--up-list` can be found in `data/pdb_classification`.